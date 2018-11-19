#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  Rigid alignment algorithm implementation based on SDF-2-SDF paper.
#  ================================================================

# stdlib
from enum import Enum
from inspect import currentframe, getframeinfo

# common libs
import numpy as np
import os.path

from math_utils.convolution import convolve_with_kernel_preserve_zeros
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
from interpolation import interpolate_warped_live, get_and_print_interpolation_data
from level_set_term import level_set_term_at_location
import smoothing_term as st
from transformation import twist_vector_to_matrix
from sdf_gradient_resp_to_twist import sdf_gradient_resp_to_twist


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
        # self.convergence_status = cpp_extension.ConvergenceStatus()


class Sdf2Sdf2d:
    def __init__(self, out_path="out2D",
                 field_size=128,
                 # TODO writers should be initialized only after the field size becomes known during optimization and
                 #  should be destroyed afterward
                 default_value=1.0,  # TODO fix default at 1.0: it should not vary

                 # adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                 gradient_descent_rate=0.1,

                 maximum_warp_length_lower_threshold=0.1,
                 maximum_warp_length_upper_threshold=10000,
                 max_iterations=100, min_iterations=1,

                 enable_component_fields=False,
                 view_scaling_factor=8):
        self.field_size = field_size
        self.out_path = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # energy
        self.total_energy = 0.

        # self.compute_method = compute_method

        # optimization parameters

        self.gradient_descent_rate = gradient_descent_rate
        self.maximum_warp_length_lower_threshold = maximum_warp_length_lower_threshold
        self.maximum_warp_length_upper_threshold = maximum_warp_length_upper_threshold
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        # self.adaptive_learning_rate_method = adaptive_learning_rate_method
        self.default_value = default_value

        """
        TODO: plotting and logging should be a separate concern performed by a different, decoupled, and 
        completely independent class
        """

        # visualization flags & parameters
        self.enable_convergence_status_logging = True
        self.enable_3d_plot = False
        self.enable_warp_quiverplot = True
        self.enable_gradient_quiverplot = True
        self.enable_component_fields = enable_component_fields
        self.view_scaling_factor = view_scaling_factor

        # statistical aggregates
        self.focus_neighborhood_log = None
        self.log = None

        # video writers
        self.live_video_writer2D = None
        self.warp_magnitude_video_writer2D = None
        self.data_gradient_video_writer2D = None
        self.smoothing_gradient_video_writer2D = None
        self.level_set_gradient_video_writer2D = None
        self.live_video_writer3D = None
        self.warp_video_writer2D = None
        self.gradient_video_writer2D = None

        # TODO: writers & other things depending on a single optimization run need to be initialized in the beginning
        # of the optimization routine (and torn down at the end of it, instead of in the object destructor)
        # initializations
        self.edasg_field = None
        self.__initialize_writers(field_size)

    def sdf_2_sdf(self, live_field, canonical_field, eta):
        final_live_field = np.copy(live_field)

        twist = np.zeros((3, 1))
        ref_weight, cur_weight = 1, 1
        for iter in range(15):
            matrix_A = np.zeros((3, 3))
            vector_b = np.zeros((3, 1))
            error = 0.
            for i in range(canonical_field.shape[1]):
                for j in range(canonical_field.shape[0]):
                    ref_sdf = canonical_field[j, i]
                    if ref_sdf < -eta:
                        ref_weight = 0
                    # apply twist to live_field
                    cur_idx = np.delete(np.dot(twist_vector_to_matrix(twist), np.array([[i], [j], [1]])), 2)
                    cur_sdf = live_field[int(cur_idx[1]), int(cur_idx[0])]
                    if cur_sdf < -eta:
                        cur_weight = 0
                    final_live_field[j, i] = cur_sdf

                    if ref_sdf < -1 or cur_sdf < -1:
                        continue
                    cur_gradient = sdf_gradient_resp_to_twist(live_field, j, i, twist)
                    matrix_A += np.dot(cur_gradient.T, cur_gradient)
                    vector_b += (ref_sdf - cur_sdf + np.dot(cur_gradient, twist)) * cur_gradient.T
                    error += (ref_sdf * ref_weight - cur_sdf * cur_weight) ** 2

            twist_star = np.dot(np.linalg.inv(matrix_A), vector_b)  # optimal solution
            twist += .5 * np.subtract(twist_star, twist)

        make_warp_vector_plot(self.warp_video_writer2D, live_field, iteration_number=15)
            # print("error(/E_geom): ", error, "at iteration ", iter)
            # print("twist vector is \n", twist)
        return final_live_field, twist, error

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
