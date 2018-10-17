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
import importlib.machinery
from inspect import currentframe, getframeinfo

# common libs
import numpy as np
import os.path

from sobolev_smoothing import convolve_with_sobolev_smoothing_kernel
import cv2

# local
from utils.tsdf_set_routines import set_zeros_for_values_outside_narrow_band_union, voxel_is_outside_narrow_band_union
from utils.vizualization import make_3d_plots, make_warp_vector_plot, warp_field_to_heatmap, \
    sdf_field_to_image, visualize_and_save_sdf_and_warp_magnitude_progression, \
    visualzie_and_save_energy_and_max_warp_progression
from utils.point import Point
from utils.printing import *
from utils.sampling import focus_coordinates_match, get_focus_coordinates
from utils.tsdf_set_routines import value_outside_narrow_band
from interpolation import interpolate_warped_live
import data_term as dt
from level_set_term import level_set_term_at_location
import smoothing_term as st

# C++ extension

# import  level_set_fusion_optimization as cpp_extension

cpp_extension = \
    importlib.machinery.ExtensionFileLoader(
        "level_set_fusion_optimization",
        "../cpp/cmake-build-release/" +
        "level_set_fusion_optimization.cpython-35m-x86_64-linux-gnu.so").load_module()


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


class ComputeMethod(Enum):
    DIRECT = 0
    VECTORIZED = 1


class Optimizer2D:
    def __init__(self, out_path="out2D",
                 field_size=128,
                 # TODO writers should be initialized only after the field size becomes known during optimization and
                 #  should be destroyed afterward
                 default_value=1.0,

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

                 maximum_warp_length_lower_threshold=0.1,
                 maximum_warp_length_upper_threshold=10000,
                 max_iterations=100, min_iterations=1,

                 sobolev_kernel=None,

                 enable_component_fields=False,
                 view_scaling_factor=8):

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

        # visualization flags & parameters
        self.enable_3d_plot = False
        self.enable_warp_quiverplot = True
        self.enable_gradient_quiverplot = True
        self.enable_component_fields = enable_component_fields
        self.view_scaling_factor = view_scaling_factor

        # statistical aggregates
        self.focus_neighborhood_log = \
            self.__generate_initial_focus_neighborhood_log()
        self.log = OptimizationLog()

        # video writers
        self.live_video_writer2D = None
        self.warp_magnitude_video_writer2D = None
        self.data_gradient_video_writer2D = None
        self.smoothing_gradient_video_writer2D = None
        self.level_set_gradient_video_writer2D = None
        self.live_video_writer3D = None
        self.warp_video_writer2D = None
        self.gradient_video_writer2D = None

        # initializations
        self.edasg_field = None
        self.__initialize_writers(field_size)
        self.last_run_iteration_count = 0

    @staticmethod
    def __run_checks(warped_live_field, canonical_field, warp_field):
        if warped_live_field.shape != canonical_field.shape or warped_live_field.shape[0] != warp_field.shape[0] or \
                warped_live_field.shape[1] != warp_field.shape[1] or warp_field.shape[0] != warp_field.shape[1]:
            raise ValueError(
                "warp field, warped live field, and canonical field all need to be 1D arrays of the same size.")

    def __optimization_iteration_vectorized(self, warped_live_field, canonical_field, warp_field,
                                            data_component_field=None, smoothing_component_field=None,
                                            level_set_component_field=None, band_union_only=True):

        live_gradient_y, live_gradient_x = np.gradient(warped_live_field)
        data_gradient_field = dt.compute_data_term_gradient_vectorized(warped_live_field, canonical_field,
                                                                       live_gradient_x, live_gradient_y)
        self.total_data_energy = \
            dt.compute_data_term_energy_contribution(warped_live_field, canonical_field) * self.data_term_weight
        smoothing_gradient_field = st.compute_smoothing_term_gradient_vectorized(warp_field)
        self.total_smoothing_energy = \
            st.compute_smoothing_term_energy(warp_field, warped_live_field,
                                             canonical_field) * self.smoothing_term_weight

        if data_component_field is not None:
            np.copyto(data_component_field, data_gradient_field)
        if smoothing_component_field is not None:
            np.copyto(smoothing_component_field, smoothing_gradient_field)
        if level_set_component_field is not None:
            frame_info = getframeinfo(currentframe())
            print("Warning: level set term not implemented in vectorized version, "
                  "passed level_set_component_field is not None, {:s} : {:d}".format(frame_info.filename,
                                                                                     frame_info.lineno))

        gradient_field = self.data_term_weight * data_gradient_field + \
                         self.smoothing_term_weight * smoothing_gradient_field

        if band_union_only:
            set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, gradient_field)

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
            convolve_with_sobolev_smoothing_kernel(gradient_field, self.sobolev_kernel)

        np.copyto(warp_field, -gradient_field * self.gradient_descent_rate)
        warp_lengths = np.linalg.norm(warp_field, axis=2)
        maximum_warp_length_at = np.unravel_index(np.argmax(warp_lengths), warp_lengths.shape)
        maximum_warp_length = warp_lengths[maximum_warp_length_at]

        # ***
        print(" Warp: ", BOLD_GREEN, warp_field[focus], RESET, " Warp length: ", BOLD_GREEN,
              np.linalg.norm(warp_field[focus]), RESET, sep='')
        # ***

        u_vectors = warp_field[:, :, 0].copy()
        v_vectors = warp_field[:, :, 1].copy()

        out_warped_live_field, (out_u_vectors, out_v_vectors) = \
            cpp_extension.interpolate(warped_live_field, canonical_field, u_vectors, v_vectors)

        np.copyto(warped_live_field, out_warped_live_field)

        # some entries might have been erased due to things in the live sdf becoming truncated
        warp_field[:, :, 0] = out_u_vectors
        warp_field[:, :, 0] = out_v_vectors

        return maximum_warp_length, maximum_warp_length_at

    def __optimization_iteration_direct(self, warped_live_field, canonical_field, warp_field, gradient_field,
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
                canonical_sdf = canonical_field[y, x]

                live_is_truncated = value_outside_narrow_band(live_sdf)
                canonical_is_truncated = value_outside_narrow_band(canonical_sdf)

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

                gradient_field[y, x] = gradient

        if self.sobolev_smoothing_enabled:
            convolve_with_sobolev_smoothing_kernel(gradient_field, self.sobolev_kernel)

        max_warp = 0.0
        max_warp_location = -1

        # update the warp field based on the gradient
        for y in range(0, field_size):
            for x in range(0, field_size):
                warp_field[y, x] = -gradient_field[y, x] * self.gradient_descent_rate
                if focus_coordinates_match(x, y):
                    print(" Warp: ", BOLD_GREEN, warp_field[y, x], RESET, " Warp length: ", BOLD_GREEN,
                          np.linalg.norm(warp_field[y, x]), RESET, sep='')
                warp_length = np.linalg.norm(warp_field[y, x])
                if warp_length > max_warp:
                    max_warp = warp_length
                    max_warp_location = Point(x, y)
                if (x, y) in self.focus_neighborhood_log:
                    log = self.focus_neighborhood_log[(x, y)]
                    log.warp_magnitudes.append(warp_length)
                    log.sdf_values.append(warped_live_field[y, x])

        # warp live frame using the warp field
        interpolate_warped_live(canonical_field, warped_live_field, warp_field, gradient_field,
                                band_union_only=False, known_values_only=False, substitute_original=False)

        # log energy aggregates
        self.log.max_warps.append(max_warp)
        self.log.data_energies.append(self.total_data_energy)
        self.log.smoothing_energies.append(self.total_smoothing_energy)
        self.log.level_set_energies.append(self.total_level_set_energy)

        return max_warp, max_warp_location,

    def optimize(self, live_field, canonical_field, warp_field):
        max_warp = np.inf
        iteration_number = 0
        gradient_field = np.zeros_like(warp_field)

        self.__run_checks(live_field, canonical_field, warp_field)

        if self.adaptive_learning_rate_method == AdaptiveLearningRateMethod.RMS_PROP:
            # exponentially decaying average of squared gradients
            self.edasg_field = np.zeros_like(live_field)

        # prepare to log fields for warp vector components from various terms if necessary
        if self.enable_component_fields:
            data_component_field = np.zeros_like(warp_field)
            smoothing_component_field = np.zeros_like(warp_field)
            if self.level_set_term_enabled:
                level_set_component_field = np.zeros_like(warp_field)
            else:
                level_set_component_field = None
        else:
            data_component_field = None
            smoothing_component_field = None
            level_set_component_field = None

        # do some logging initialization that requires canonical data
        for (x, y), log in self.focus_neighborhood_log.items():
            log.canonical_sdf = canonical_field[y, x]

        # write original raw live
        if self.live_video_writer3D is not None:
            make_3d_plots(self.live_video_writer3D, canonical_field, live_field, warp_field)
        if self.live_video_writer2D is not None:
            self.live_video_writer2D.write(sdf_field_to_image(live_field, self.view_scaling_factor))

        # actually perform the optimization
        while (iteration_number < self.min_iterations) or \
                (iteration_number < self.max_iterations and
                 self.maximum_warp_length_lower_threshold < max_warp < self.maximum_warp_length_upper_threshold):

            if self.compute_method == ComputeMethod.DIRECT:
                max_warp, max_warp_location = \
                    self.__optimization_iteration_direct(live_field, canonical_field, warp_field, gradient_field,
                                                         data_component_field, smoothing_component_field,
                                                         level_set_component_field)
            elif self.compute_method == ComputeMethod.VECTORIZED:
                max_warp, max_warp_location = \
                    self.__optimization_iteration_vectorized(live_field, canonical_field, warp_field, gradient_field,
                                                             data_component_field, smoothing_component_field,
                                                             level_set_component_field)

            level_set_energy_string = ""
            if self.level_set_term_enabled:
                level_set_energy_string = "; level set energy: {:5f}".format(self.total_level_set_energy)

            print(BOLD_RED, "[Iteration ", iteration_number, " done],", RESET,
                  " data energy: {:5f}".format(self.total_data_energy),
                  "; smoothing energy: {:5f}".format(self.total_smoothing_energy), level_set_energy_string,
                  "; total energy:", self.total_data_energy + self.total_smoothing_energy + self.total_level_set_energy,
                  "; max warp:", max_warp, "@", max_warp_location, sep="")

            self.__make_iteration_visualizations(iteration_number, warp_field, gradient_field, data_component_field,
                                                 smoothing_component_field, level_set_component_field, live_field,
                                                 canonical_field)

            iteration_number += 1
        self.last_run_iteration_count = iteration_number

    def __del__(self):
        if self.live_video_writer3D is not None:
            self.live_video_writer3D.release()
        if self.warp_video_writer2D is not None:
            self.warp_video_writer2D.release()
        if self.gradient_video_writer2D is not None:
            self.gradient_video_writer2D.release()
        if self.data_gradient_video_writer2D is not None:
            self.data_gradient_video_writer2D.release()
        if self.smoothing_gradient_video_writer2D is not None:
            self.smoothing_gradient_video_writer2D.release()
        if self.level_set_gradient_video_writer2D is not None:
            self.level_set_gradient_video_writer2D.release()

        self.live_video_writer2D.release()
        self.warp_magnitude_video_writer2D.release()

    def __initialize_writers(self, field_size):
        self.live_video_writer2D = cv2.VideoWriter(
            os.path.join(self.out_path, 'live_field_evolution_2D.mkv'),
            cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
            (field_size * self.view_scaling_factor, field_size * self.view_scaling_factor),
            isColor=False)
        self.warp_magnitude_video_writer2D = cv2.VideoWriter(
            os.path.join(self.out_path, 'warp_magnitudes_2D.mkv'),
            cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
            (field_size * self.view_scaling_factor, field_size * self.view_scaling_factor),
            isColor=True)
        if self.enable_3d_plot:
            self.live_video_writer3D = cv2.VideoWriter(
                os.path.join(self.out_path, 'live_field_evolution_2D_3D_plot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1230, 720), isColor=True)
        if self.enable_warp_quiverplot:
            self.warp_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'warp_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
        if self.enable_gradient_quiverplot:
            self.gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'gradient_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
        if self.enable_component_fields:
            self.data_gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'data_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
            self.smoothing_gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'smoothing_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
            if self.level_set_term_enabled:
                self.level_set_gradient_video_writer2D = cv2.VideoWriter(
                    os.path.join(self.out_path, 'level_set_2D_quiverplot.mkv'),
                    cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)

    def plot_logged_sdf_and_warp_magnitudes(self):
        visualize_and_save_sdf_and_warp_magnitude_progression(get_focus_coordinates(),
                                                              self.focus_neighborhood_log,
                                                              self.out_path)

    def plot_logged_energies_and_max_warps(self):
        visualzie_and_save_energy_and_max_warp_progression(self.log, self.out_path)

    @staticmethod
    def __generate_initial_focus_neighborhood_log():
        focus_coordinates = get_focus_coordinates()
        neighborhood_log = {}
        for y in range(focus_coordinates[1] - 1, focus_coordinates[1] + 2):
            for x in range(focus_coordinates[0] - 1, focus_coordinates[0] + 2):
                neighborhood_log[(x, y)] = VoxelLog()

        return neighborhood_log

    def __make_iteration_visualizations(self, iteration_number, warp_field, gradient_field, data_component_field,
                                        smoothing_component_field, level_set_component_field, live_field,
                                        canonical_field):
        if self.warp_video_writer2D is not None:
            make_warp_vector_plot(self.warp_video_writer2D, warp_field,
                                  scale=10.0, iteration_number=iteration_number,
                                  vectors_name="Warp vectors (scaled x10)")
        if self.gradient_video_writer2D is not None:
            make_warp_vector_plot(self.gradient_video_writer2D, -gradient_field, iteration_number=iteration_number,
                                  vectors_name="Gradient vectors (negated)")
        if self.data_gradient_video_writer2D is not None:
            make_warp_vector_plot(self.data_gradient_video_writer2D, -data_component_field,
                                  iteration_number=iteration_number, vectors_name="Data gradients (negated)")
        if self.smoothing_gradient_video_writer2D is not None:
            make_warp_vector_plot(self.smoothing_gradient_video_writer2D, -smoothing_component_field,
                                  iteration_number=iteration_number, vectors_name="Smoothing gradients (negated)")
        if self.level_set_gradient_video_writer2D is not None:
            make_warp_vector_plot(self.level_set_gradient_video_writer2D, -level_set_component_field,
                                  iteration_number=iteration_number, vectors_name="Level set gradients (negated)")

        if self.live_video_writer2D is not None:
            self.live_video_writer2D.write(sdf_field_to_image(live_field, self.view_scaling_factor))
        if self.warp_magnitude_video_writer2D is not None:
            self.warp_magnitude_video_writer2D.write(warp_field_to_heatmap(warp_field, self.view_scaling_factor))
        if self.live_video_writer3D is not None:
            make_3d_plots(self.live_video_writer3D, canonical_field, live_field, warp_field)
