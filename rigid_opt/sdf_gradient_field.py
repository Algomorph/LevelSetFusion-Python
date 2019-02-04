#  ================================================================
#  Created by Fei Shan on 02/01/19.
#  Calculate sdf gradient w.r.t. transformation vector
#  ================================================================

# common libs
import numpy as np

# local
from rigid_opt.transformation import twist_vector_to_matrix


class GradientField:
    def __init__(self):
        """
        Constructor
        """

    def calculate(self, live_field, twist, array_offset, voxel_size=0.004):
        gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)
        sdf_gradient_first_term = np.gradient(live_field)
        twist_matrix_homo_inv = twist_vector_to_matrix(-twist)

        y_voxel = 0.0
        w_voxel = 1.0

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                x_voxel = (x_field + array_offset[0]) * voxel_size
                z_voxel = (y_field + array_offset[2]) * voxel_size  # acts as "Z" coordinate

                point = np.array([[x_voxel, z_voxel, w_voxel]], dtype=np.float32).T
                trans = np.dot(twist_matrix_homo_inv, point)

                sdf_gradient_second_term = np.array([[1, 0, trans[1]],
                                                     [0, 1, -trans[0]]])
                gradient_field[y_field, x_field] = np.dot(np.array([sdf_gradient_first_term[1][y_field,
                                                                                               x_field]/voxel_size,
                                                                    sdf_gradient_first_term[0][y_field,
                                                                                               x_field]/voxel_size]),
                                                          sdf_gradient_second_term)
                # gradient_field[i, j] = self.sdf_gradient_wrt_twist(live_field, i, j, twist_vector)
        return gradient_field

    # def sdf_gradient_wrt_twist(self, live_field, voxel_x, voxel_z, twist_vector):
    #     # sdf_gradient_wrt_voxel = np.array([np.gradient(live_field)[1][voxel_z, voxel_x],
    #     #                                       -np.gradient(live_field)[0][voxel_z, voxel_x]])
    #     sdf_gradient_wrt_voxel = np.zeros((1, 2))
    #
    #     if voxel_x - 1 < 0:
    #         post_sdf = live_field[voxel_x + 1, voxel_z]
    #         if post_sdf < -1:
    #             sdf_gradient_wrt_voxel[0, 0] = 0
    #         else:
    #             sdf_gradient_wrt_voxel[0, 0] = post_sdf - live_field[voxel_x, voxel_z]
    #     elif voxel_x + 1 > live_field.shape[0] - 1:
    #         pre_sdf = live_field[voxel_x - 1, voxel_z]
    #         if pre_sdf < -1:
    #             sdf_gradient_wrt_voxel[0, 0] = 0
    #         else:
    #             sdf_gradient_wrt_voxel[0, 0] = live_field[voxel_x, voxel_z] - pre_sdf
    #     else:
    #         pre_sdf = live_field[voxel_x - 1, voxel_z]
    #         post_sdf = live_field[voxel_x + 1, voxel_z]
    #         if (post_sdf < -1) or (pre_sdf < -1):
    #             sdf_gradient_wrt_voxel[0, 0] = 0
    #         else:
    #             sdf_gradient_wrt_voxel[0, 0] = (post_sdf - pre_sdf) / 2
    #
    #     if voxel_z - 1 < 0:
    #         post_sdf = live_field[voxel_x, voxel_z + 1]
    #         if post_sdf < -1:
    #             sdf_gradient_wrt_voxel[0, 1] = 0
    #         else:
    #             sdf_gradient_wrt_voxel[0, 1] = post_sdf - live_field[voxel_x, voxel_z]
    #     elif voxel_z + 1 > live_field.shape[1] - 1:
    #         pre_sdf = live_field[voxel_x, voxel_z - 1]
    #         if pre_sdf < -1:
    #             sdf_gradient_wrt_voxel[0, 1] = 0
    #         else:
    #             sdf_gradient_wrt_voxel[0, 1] = live_field[voxel_x, voxel_z] - pre_sdf
    #     else:
    #         pre_sdf = live_field[voxel_x, voxel_z - 1]
    #         post_sdf = live_field[voxel_x, voxel_z + 1]
    #         if (post_sdf < -1) or (pre_sdf < -1):
    #             sdf_gradient_wrt_voxel[0, 1] = 0
    #         else:
    #             sdf_gradient_wrt_voxel[0, 1] = (post_sdf - pre_sdf) / 2
    #
    #     twist_matrix_homo = twist_vector_to_matrix(twist_vector)
    #     voxel_transformed_homo = np.dot(np.linalg.inv(twist_matrix_homo), np.array([[voxel_x], [voxel_z], [1]]))
    #     voxel_transformed = np.delete(voxel_transformed_homo, 2)
    #     voxel_gradient_wrt_twist = np.concatenate((np.identity(2),
    #                                                    np.array([[-voxel_transformed[1]], [voxel_transformed[0]]])),
    #                                                    axis=1)
    #
    #     return np.dot(sdf_gradient_wrt_voxel, voxel_gradient_wrt_twist).reshape((1, -1))
