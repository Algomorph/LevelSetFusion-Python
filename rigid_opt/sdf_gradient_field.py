#  ================================================================
#  Created by Fei Shan on 02/01/19.
#  Calculate sdf gradient w.r.t. transformation vector
#  ================================================================

# common libs
import numpy as np

# local
from math_utils.transformation import twist_vector_to_matrix2d


def calculate_gradient_wrt_twist(live_field, twist, array_offset, voxel_size=0.004):
    gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)
    sdf_gradient_first_term = np.gradient(live_field)
    twist_matrix_inv = twist_vector_to_matrix2d(-twist)

    y_voxel = 0.0
    w_voxel = 1.0

    for y_field in range(live_field.shape[0]):
        for x_field in range(live_field.shape[1]):
            x_voxel = (x_field + array_offset[0]) * voxel_size
            z_voxel = (y_field + array_offset[2]) * voxel_size  # acts as "Z" coordinate

            point = np.array([[x_voxel, z_voxel, w_voxel]], dtype=np.float32).T
            trans = np.dot(twist_matrix_inv, point)

            sdf_gradient_second_term = np.array([[1, 0, trans[1]],
                                                 [0, 1, -trans[0]]])
            gradient_field[y_field, x_field] = np.dot(np.array([sdf_gradient_first_term[1][y_field, x_field],
                                                                sdf_gradient_first_term[0][y_field, x_field]]),
                                                      sdf_gradient_second_term)
    gradient_field /= voxel_size


    return gradient_field