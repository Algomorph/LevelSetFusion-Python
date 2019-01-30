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
from rigid_opt.transformation import twist_vector_to_matrix, affine_of_voxel2d
from rigid_opt.sdf_gradient_wrt_twist import GradientField
import utils.printing as printing
from rigid_opt.sdf_2_sdf_visualizer import Sdf2SdfVisualizer


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
                 depth_image1d,
                 canonical_field,
                 live_field,
                 camera,
                 offset,
                 eta=.01,
                 iteration=60,
                 voxel_size=0.004
                 ):
        """
        Optimization algorithm
        :param canonical_field:
        :param live_field:
        :param eta: thickness of surface, used to determine reliability of sdf field
        :param iteration: total number of iterations
        :return:
        """

        final_canonical_field = np.copy(canonical_field)
        canonical_weight = np.ones_like(canonical_field)
        live_weight = np.ones_like(live_field)

        twist = np.zeros((3, 1))
        for iteration_count in range(iteration):
            matrix_a = np.zeros((3, 3))
            vector_b = np.zeros((3, 1))
            # energy = 0.  # Energy term
            canonical_weight = (canonical_field > -eta).astype(np.int)
            live_field = affine_of_voxel2d(live_field, twist, depth_image1d, camera, offset, voxel_size)
            live_weight = (live_field > -eta).astype(np.int)
            live_gradient = GradientField().calculate(live_field, twist)
            # print(live_gradient.min(), live_gradient.max())
            for x in range(live_field.shape[0]):
                for z in range(live_field.shape[1]):
                    matrix_a += np.dot(live_gradient[x, z][:, None], live_gradient[x, z][None, :])
                    vector_b += (canonical_field[x, z] - live_field[x, z] +
                                 np.dot(live_gradient[x, z].reshape((1, -1)), twist)) \
                                * live_gradient[x, z].reshape((1, -1)).T
            energy = np.sum((canonical_field * canonical_weight - live_field * live_weight) ** 2)
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
