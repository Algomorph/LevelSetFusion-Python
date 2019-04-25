#  ================================================================
#  Created by Gregory Kramida on 9/18/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

# stdlib
from enum import Enum
from inspect import currentframe, getframeinfo

# common libs
import numpy as np
import os.path

from math_utils.convolution import convolve_with_kernel_preserve_zeros

# local
from utils.tsdf_set_routines import set_zeros_for_values_outside_narrow_band_union, voxel_is_outside_narrow_band_union
from utils.visualization import visualize_and_save_sdf_and_warp_magnitude_progression, \
    visualzie_and_save_energy_and_max_warp_progression
from utils.point2d import Point2d
from utils.printing import *
from utils.sampling import focus_coordinates_match, get_focus_coordinates
from utils.tsdf_set_routines import value_outside_narrow_band
from nonrigid_opt.field_warping import warp_field_advanced, get_and_print_interpolation_data
from nonrigid_opt.slavcheva.level_set_term import level_set_term_at_location
from nonrigid_opt.slavcheva import data_term as dt, smoothing_term as st, slavcheva_visualizer as viz

# C++ extension
import level_set_fusion_optimization as cpp


class AdaptiveLearningRateMethod(Enum):
    NONE = 0
    RMS_PROP = 1


class VoxelLog:
    def __init__(self):
        self.warp_magnitudes = []
        self.sdf_values = []
        self.canonical_sdf = 0.0

    def __repr__(self):
        return str(self.warp_magnitudes) + "; " + str(self.sdf_values)


class OptimizationLog:
    def __init__(self):
        self.data_energies = []
        self.smoothing_energies = []
        self.level_set_energies = []
        self.max_warps = []
        self.convergence_report = cpp.ConvergenceReport2d()


class ComputeMethod(Enum):
    DIRECT = 0
    VECTORIZED = 1


class SlavchevaOptimizer2d:

    def __init__(self, out_path="out2D",
                 field_size=128,
                 # TODO writers should be initialized only after the field size becomes known during optimization and
                 #  should be destroyed afterward
                 default_value=1.0,  # TODO fix default at 1.0: it should not vary

                 compute_method=ComputeMethod.DIRECT,

                 level_set_term_enabled=False,
                 sobolev_smoothing_enabled=False,

                 data_term_method=dt.DataTermMethod.BASIC,
                 smoothing_term_method=st.SmoothingTermMethod.TIKHONOV,
                 adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                 gradient_descent_rate=0.1,
                 data_term_weight=1.0,
                 smoothing_term_weight=0.2,
                 isomorphic_enforcement_factor=0.1,
                 level_set_term_weight=0.2,

                 maximum_warp_length_lower_threshold=0.1,  # in terms of voxel size
                 maximum_warp_length_upper_threshold=10000,
                 max_iterations=100, min_iterations=1,

                 sobolev_kernel=None,
                 visualization_settings=None,
                 enable_convergence_status_logging=True
                 ):

        if visualization_settings:
            self.visualization_settings = visualization_settings
        else:
            self.visualization_settings = viz.SlavchevaVisualizer.Settings()
        self.visualizer = None

        self.field_size = field_size
        self.out_path = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # energies
        self.total_data_energy = 0.
        self.total_smoothing_energy = 0.
        self.total_level_set_energy = 0.

        self.compute_method = compute_method

        # optimization parameters
        self.level_set_term_enabled = level_set_term_enabled
        self.sobolev_smoothing_enabled = sobolev_smoothing_enabled

        self.gradient_descent_rate = gradient_descent_rate
        self.data_term_weight = data_term_weight
        self.smoothing_term_weight = smoothing_term_weight
        self.isomorphic_enforcement_factor = isomorphic_enforcement_factor
        self.level_set_term_weight = level_set_term_weight
        self.maximum_warp_length_lower_threshold = maximum_warp_length_lower_threshold
        self.maximum_warp_length_upper_threshold = maximum_warp_length_upper_threshold
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.sobolev_kernel = sobolev_kernel

        self.data_term_method = data_term_method
        self.smoothing_term_method = smoothing_term_method
        self.adaptive_learning_rate_method = adaptive_learning_rate_method
        self.default_value = default_value

        """
        TODO: plotting and logging should be a separate concern performed by a different, decoupled, and 
        completely independent class
        """

        # logging and statistical aggregates
        self.focus_neighborhood_log = None
        self.log = None
        self.enable_convergence_status_logging = enable_convergence_status_logging

        self.gradient_field = None
        # adaptive learning rate
        self.edasg_field = None

    @staticmethod
    def __run_checks(warped_live_field, canonical_field, warp_field):
        if warped_live_field.shape != canonical_field.shape or warped_live_field.shape[0] != warp_field.shape[0] or \
                warped_live_field.shape[1] != warp_field.shape[1] or warp_field.shape[0] != warp_field.shape[1]:
            raise ValueError(
                "warp field, warped live field, and canonical field all need to be 1D arrays of the same size.")

    def __optimization_iteration_vectorized(self, warped_live_field, canonical_field, warp_field, band_union_only=True):

        live_gradient_y, live_gradient_x = np.gradient(warped_live_field)
        data_gradient_field = dt.compute_data_term_gradient_vectorized(warped_live_field, canonical_field,
                                                                       live_gradient_x, live_gradient_y)
        set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, data_gradient_field)
        self.total_data_energy = \
            dt.compute_data_term_energy_contribution(warped_live_field, canonical_field) * self.data_term_weight
        smoothing_gradient_field = st.compute_smoothing_term_gradient_vectorized(warp_field)
        self.total_smoothing_energy = \
            st.compute_smoothing_term_energy(warp_field, warped_live_field,
                                             canonical_field) * self.smoothing_term_weight

        if self.visualizer.data_component_field is not None:
            np.copyto(self.visualizer.data_component_field, data_gradient_field)
        if self.visualizer.smoothing_component_field is not None:
            np.copyto(self.visualizer.smoothing_component_field, smoothing_gradient_field)
        if self.visualizer.level_set_component_field is not None:
            frame_info = getframeinfo(currentframe())
            print("Warning: level set term not implemented in vectorized version, "
                  "passed level_set_component_field is not None, {:s} : {:d}".format(frame_info.filename,
                                                                                     frame_info.lineno))

        self.gradient_field = self.data_term_weight * data_gradient_field + \
                              self.smoothing_term_weight * smoothing_gradient_field

        if band_union_only:
            set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, self.gradient_field)

        # *** Print information at focus voxel
        focus_x, focus_y = get_focus_coordinates()
        focus = (focus_y, focus_x)
        print("Point: ", focus_x, ",", focus_y, sep='', end='')
        dt.compute_local_data_term(warped_live_field, canonical_field, focus_x, focus_y, live_gradient_x,
                                   live_gradient_y, method=dt.DataTermMethod.BASIC)
        focus_data_gradient = data_gradient_field[focus]
        print(" Data grad: ", BOLD_GREEN, -focus_data_gradient, RESET, sep='', end='')

        st.compute_local_smoothing_term_gradient(warp_field, focus_x, focus_y, method=self.smoothing_term_method,
                                                 copy_if_zero=False,
                                                 isomorphic_enforcement_factor=self.isomorphic_enforcement_factor)
        focus_smoothing_gradient = smoothing_gradient_field[focus] * self.smoothing_term_weight
        print(" Smoothing grad (scaled): ", BOLD_GREEN,
              -focus_smoothing_gradient, RESET, sep='', end='')

        # ***
        if self.sobolev_smoothing_enabled:
            convolve_with_kernel_preserve_zeros(self.gradient_field, self.sobolev_kernel, True)

        np.copyto(warp_field, -self.gradient_field * self.gradient_descent_rate)
        warp_lengths = np.linalg.norm(warp_field, axis=2)
        maximum_warp_length_at = np.unravel_index(np.argmax(warp_lengths), warp_lengths.shape)
        maximum_warp_length = warp_lengths[maximum_warp_length_at]

        # ***
        print(" Warp: ", BOLD_GREEN, warp_field[focus], RESET, " Warp length: ", BOLD_GREEN,
              np.linalg.norm(warp_field[focus]), RESET, sep='')
        # ***

        get_and_print_interpolation_data(canonical_field, warped_live_field, warp_field, focus_x, focus_y)

        u_vectors = warp_field[:, :, 0].copy()
        v_vectors = warp_field[:, :, 1].copy()

        out_warped_live_field, (out_u_vectors, out_v_vectors) = \
            cpp.warp_field_advanced(warped_live_field, canonical_field, u_vectors, v_vectors)

        np.copyto(warped_live_field, out_warped_live_field)

        # some entries might have been erased due to things in the live sdf becoming truncated
        warp_field[:, :, 0] = out_u_vectors
        warp_field[:, :, 1] = out_v_vectors

        return maximum_warp_length, Point2d(maximum_warp_length_at[1], maximum_warp_length_at[0])

    def __optimization_iteration_direct(self, warped_live_field, canonical_field, warp_field,
                                        data_component_field=None, smoothing_component_field=None,
                                        level_set_component_field=None, band_union_only=True):

        self.total_data_energy = 0.
        self.total_smoothing_energy = 0.
        self.total_level_set_energy = 0.

        field_size = warp_field.shape[0]

        live_gradient_y, live_gradient_x = np.gradient(warped_live_field)

        for y in range(0, field_size):
            for x in range(0, field_size):
                if focus_coordinates_match(x, y):
                    print("Point: ", x, ",", y, sep='', end='')

                gradient = 0.0

                live_sdf = warped_live_field[y, x]

                live_is_truncated = value_outside_narrow_band(live_sdf)

                if band_union_only and voxel_is_outside_narrow_band_union(warped_live_field, canonical_field, x, y):
                    continue

                data_gradient, local_data_energy = \
                    dt.compute_local_data_term(warped_live_field, canonical_field, x, y, live_gradient_x,
                                               live_gradient_y, method=self.data_term_method)
                scaled_data_gradient = self.data_term_weight * data_gradient
                self.total_data_energy += self.data_term_weight * local_data_energy
                gradient += scaled_data_gradient
                if focus_coordinates_match(x, y):
                    print(" Data grad: ", BOLD_GREEN, -data_gradient, RESET, sep='', end='')
                if data_component_field is not None:
                    data_component_field[y, x] = data_gradient
                if self.level_set_term_enabled and not live_is_truncated:
                    level_set_gradient, local_level_set_energy = \
                        level_set_term_at_location(warped_live_field, x, y)
                    scaled_level_set_gradient = self.level_set_term_weight * level_set_gradient
                    self.total_level_set_energy += self.level_set_term_weight * local_level_set_energy
                    gradient += scaled_level_set_gradient
                    if level_set_component_field is not None:
                        level_set_component_field[y, x] = level_set_gradient
                    if focus_coordinates_match(x, y):
                        print(" Level-set grad (scaled): ", BOLD_GREEN,
                              -scaled_level_set_gradient, RESET, sep='', end='')

                smoothing_gradient, local_smoothing_energy = \
                    st.compute_local_smoothing_term_gradient(warp_field, x, y, method=self.smoothing_term_method,
                                                             copy_if_zero=False,
                                                             isomorphic_enforcement_factor=
                                                             self.isomorphic_enforcement_factor)
                scaled_smoothing_gradient = self.smoothing_term_weight * smoothing_gradient
                self.total_smoothing_energy += self.smoothing_term_weight * local_smoothing_energy
                gradient += scaled_smoothing_gradient
                if smoothing_component_field is not None:
                    smoothing_component_field[y, x] = smoothing_gradient
                if focus_coordinates_match(x, y):
                    print(" Smoothing grad (scaled): ", BOLD_GREEN,
                          -scaled_smoothing_gradient, RESET, sep='', end='')

                self.gradient_field[y, x] = gradient

        if self.sobolev_smoothing_enabled:
            convolve_with_kernel_preserve_zeros(self.gradient_field, self.sobolev_kernel, True)

        max_warp = 0.0
        max_warp_location = -1

        # update the warp field based on the gradient
        for y in range(0, field_size):
            for x in range(0, field_size):
                warp_field[y, x] = -self.gradient_field[y, x] * self.gradient_descent_rate
                if focus_coordinates_match(x, y):
                    print(" Warp: ", BOLD_GREEN, warp_field[y, x], RESET, " Warp length: ", BOLD_GREEN,
                          np.linalg.norm(warp_field[y, x]), RESET, sep='')
                warp_length = np.linalg.norm(warp_field[y, x])
                if warp_length > max_warp:
                    max_warp = warp_length
                    max_warp_location = Point2d(x, y)
                if (x, y) in self.focus_neighborhood_log:
                    log = self.focus_neighborhood_log[(x, y)]
                    log.warp_magnitudes.append(warp_length)
                    log.sdf_values.append(warped_live_field[y, x])

        new_warped_live_field = warp_field_advanced(canonical_field, warped_live_field, warp_field,
                                                    self.gradient_field,
                                                    band_union_only=False, known_values_only=False,
                                                    substitute_original=False)
        np.copyto(warped_live_field, new_warped_live_field)

        return max_warp, max_warp_location

    def optimize(self, live_field, canonical_field):

        self.visualizer = viz.SlavchevaVisualizer(len(live_field), self.out_path, self.visualization_settings)

        self.focus_neighborhood_log = \
            self.__generate_initial_focus_neighborhood_log(self.field_size)
        self.log = OptimizationLog()
        warp_field = np.zeros((self.field_size, self.field_size, 2), dtype=np.float32)

        max_warp = np.inf
        max_warp_location = (0, 0)
        iteration_number = 0
        self.gradient_field = np.zeros_like(warp_field)

        self.__run_checks(live_field, canonical_field, warp_field)

        if self.adaptive_learning_rate_method == AdaptiveLearningRateMethod.RMS_PROP:
            # exponentially decaying average of squared gradients
            self.edasg_field = np.zeros_like(live_field)

        # do some logging initialization that requires canonical data
        for (x, y), log in self.focus_neighborhood_log.items():
            log.canonical_sdf = canonical_field[y, x]

        # write original raw live
        self.visualizer.write_live_sdf_visualizations(canonical_field, live_field)

        # actually perform the optimization
        while (iteration_number < self.min_iterations) or \
                (iteration_number < self.max_iterations and
                 self.maximum_warp_length_lower_threshold < max_warp < self.maximum_warp_length_upper_threshold):

            if self.compute_method == ComputeMethod.DIRECT:
                max_warp, max_warp_location = \
                    self.__optimization_iteration_direct(live_field, canonical_field, warp_field)
            elif self.compute_method == ComputeMethod.VECTORIZED:
                max_warp, max_warp_location = \
                    self.__optimization_iteration_vectorized(live_field, canonical_field, warp_field)
            # log energy aggregates
            self.log.max_warps.append(max_warp)
            self.log.data_energies.append(self.total_data_energy)
            self.log.smoothing_energies.append(self.total_smoothing_energy)
            self.log.level_set_energies.append(self.total_level_set_energy)

            # print end-of-iteration output
            level_set_energy_string = ""
            if self.level_set_term_enabled:
                level_set_energy_string = "; level set energy: {:5f}".format(self.total_level_set_energy)

            print(BOLD_RED, "[Iteration ", iteration_number, " done],", RESET,
                  " data energy: {:5f}".format(self.total_data_energy),
                  "; smoothing energy: {:5f}".format(self.total_smoothing_energy), level_set_energy_string,
                  "; total energy:", self.total_data_energy + self.total_smoothing_energy + self.total_level_set_energy,
                  "; max warp:", max_warp, "@", max_warp_location, sep="")

            self.visualizer.write_all_iteration_visualizations(iteration_number, warp_field, self.gradient_field,
                                                               live_field, canonical_field)

            iteration_number += 1

        # log end-of-optimization stats
        if self.enable_convergence_status_logging:
            warp_stats = cpp.build_warp_delta_statistics_2d(warp_field, canonical_field, live_field,
                                                           self.maximum_warp_length_lower_threshold,
                                                           self.maximum_warp_length_upper_threshold)

            tsdf_stats = cpp.build_tsdf_difference_statistics_2d(canonical_field, live_field)

            self.log.convergence_report = cpp.ConvergenceReport2d(
                iteration_number,
                iteration_number >= self.max_iterations,
                warp_stats,
                tsdf_stats)

        del self.visualizer
        self.visualizer = None
        return live_field

    def get_convergence_report(self):
        return self.log.convergence_report

    def plot_logged_sdf_and_warp_magnitudes(self):
        visualize_and_save_sdf_and_warp_magnitude_progression(get_focus_coordinates(),
                                                              self.focus_neighborhood_log,
                                                              self.out_path)

    def plot_logged_energies_and_max_warps(self):
        visualzie_and_save_energy_and_max_warp_progression(self.log, self.out_path)

    @staticmethod
    def __generate_initial_focus_neighborhood_log(field_size):
        focus_coordinates = get_focus_coordinates()
        neighborhood_log = {}
        for y in range(focus_coordinates[1] - 1, focus_coordinates[1] + 2):
            for x in range(focus_coordinates[0] - 1, focus_coordinates[0] + 2):
                if 0 <= x < field_size and 0 <= y < field_size:
                    neighborhood_log[(x, y)] = VoxelLog()

        return neighborhood_log
