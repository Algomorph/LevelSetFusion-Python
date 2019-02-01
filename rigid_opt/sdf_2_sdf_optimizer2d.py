#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  Rigid alignment algorithm implementation based on SDF-2-SDF paper.
#  ================================================================

# stdlib
from enum import Enum
from inspect import currentframe, getframeinfo

# common libs
import numpy as np
import scipy.ndimage
import os.path
import cv2

# local
from rigid_opt.sdf_gradient_field import GradientField
import utils.printing as printing
from rigid_opt.sdf_2_sdf_visualizer import Sdf2SdfVisualizer
from rigid_opt.sdf_generation import ImageBasedSingleFrameDataset
from tsdf import generation as tsdf_gen

class Sdf2SdfOptimizer2d:
    """

    """

    class VerbosityParameters:
        """
        Parameters that controls verbosity to stdout.
        Assumes being used in an "immutable" manner, i.e. just a structure that holds values
        """

        def __init__(self, print_max_warp_update=False, print_iteration_energy=False):
            self.print_max_warp_update = print_max_warp_update
            self.print_iteration_energy = print_iteration_energy
            self.per_iteration_flags = [self.print_max_warp_update,
                                        self.print_iteration_energy]
            self.print_per_iteration_info = any(self.per_iteration_flags)

    def __init__(self,
                 verbosity_parameters=None,
                 visualization_parameters=None
                 ):
        """
        Constructor
        :param verbosity_parameters:
        :param visualization_parameters:
        """

        if verbosity_parameters:
            self.verbosity_parameters = verbosity_parameters
        else:
            self.verbosity_parameters = Sdf2SdfOptimizer2d.VerbosityParameters()

        if visualization_parameters:
            self.visualization_parameters = visualization_parameters
        else:
            self.visualization_parameters = Sdf2SdfVisualizer.Parameters()

        self.visualizer = None

    def optimize(self,
                 data_to_use,
                 eta=.01,
                 voxel_size=0.004,
                 narrow_band_width_voxels=20.,
                 iteration = 60,
                 ):
        """
        Optimization algorithm
        :param data_to_use:
        :param eta: thickness of surface, used to determine reliability of sdf field
        :param iteration: total number of iterations
        :param voxel_size: voxel side length
        :param narrow_band_width_voxels:
        :return:
        """

        canonical_field = data_to_use.generate_2d_canonical_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                                  method=tsdf_gen.DepthInterpolationMethod.NONE)
        offset = data_to_use.offset
        twist = np.zeros((3, 1))
        for iteration_count in range(iteration):
            matrix_a = np.zeros((3, 3))
            vector_b = np.zeros((3, 1))
            # energy = 0.  # Energy term
            canonical_weight = (canonical_field > -eta).astype(np.int)
            # live_field = affine_of_voxel2d(live_field, twist, depth_image1d, camera, offset, voxel_size,
            #                                camera_extrinsic_matrix=np.eye(3, dtype=np.float32),
            #                                narrow_band_width_voxels=1.)
            twist_3d = np.array([twist[0],
                                 [0.],
                                 twist[1],
                                 [0.],
                                 twist[2],
                                 [0.]], dtype=np.float32)
            live_field = data_to_use.generate_2d_live_field(narrow_band_width_voxels=narrow_band_width_voxels,
                                                            method=tsdf_gen.DepthInterpolationMethod.NONE,
                                                            apply_transformation=True, twist=twist_3d)
            live_weight = (live_field > -eta).astype(np.int)
            live_gradient = GradientField().calculate(live_field, twist, array_offset=offset, voxel_size=voxel_size)

            for i in range(live_field.shape[0]):
                for j in range(live_field.shape[1]):
                    matrix_a += np.dot(live_gradient[i, j][:, None], live_gradient[i, j][None, :])
                    vector_b += (canonical_field[i, j] - live_field[i, j] +
                                 np.dot(live_gradient[i, j][None, :], twist)) * live_gradient[i, j][:, None]

            energy = 0.5 * np.sum((canonical_field * canonical_weight - live_field * live_weight) ** 2)
            if self.verbosity_parameters.print_per_iteration_info:
                print("%s[ITERATION %d COMPLETED]%s" % (printing.BOLD_LIGHT_CYAN, iteration_count, printing.RESET),
                      end="")
                if self.verbosity_parameters.print_iteration_energy:
                    print(" energy: %f" % energy, end="")
                    print("")

            if not np.isfinite(np.linalg.cond(matrix_a)):
                print("%sSINGULAR MATRIX!%s" % (printing.BOLD_YELLOW, printing.RESET))
                continue

            twist_star = np.dot(np.linalg.inv(matrix_a), vector_b)
            twist += .5 * np.subtract(twist_star, twist)

            if self.verbosity_parameters.print_max_warp_update:
                print("optimal twist: %f, %f, %f, twist: %f, %f, %f"
                      % (twist_star[0], twist_star[1], twist_star[2], twist[0], twist[1], twist[2]), end="")
                print("")

        return

        # twist = np.zeros((3, 1))
        # ref_weight, cur_weight = 1, 1
        # for iter in range(15):
        #     matrix_A = np.zeros((3, 3))
        #     vector_b = np.zeros((3, 1))
        #     error = 0.
        #     for i in range(canonical_field.shape[1]):
        #         for j in range(canonical_field.shape[0]):
        #             ref_sdf = canonical_field[j, i]
        #             if ref_sdf < -eta:
        #                 ref_weight = 0
        #             # apply twist to live_field
        #             cur_idx = np.delete(np.dot(twist_vector_to_matrix(twist), np.array([[i], [j], [1]])), 2)
        #             cur_sdf = live_field[int(cur_idx[1]), int(cur_idx[0])]
        #             if cur_sdf < -eta:
        #                 cur_weight = 0
        #             final_live_field[j, i] = cur_sdf
        #
        #             if ref_sdf < -1 or cur_sdf < -1:
        #                 continue
        #             cur_gradient = sdf_gradient_wrt_to_twist(live_field, j, i, twist)
        #             matrix_A += np.dot(cur_gradient.T, cur_gradient)
        #             vector_b += (ref_sdf - cur_sdf + np.dot(cur_gradient, twist)) * cur_gradient.T
        #             error += (ref_sdf * ref_weight - cur_sdf * cur_weight) ** 2
        #
        #     twist_star = np.dot(np.linalg.inv(matrix_A), vector_b)  # optimal solution
        #     twist += .5 * np.subtract(twist_star, twist)
        #     print("error(/E_geom): ", error, "at iteration ", iter)
        #     print("twist vector is \n", twist)
        # return final_canonical_field, twist, error
