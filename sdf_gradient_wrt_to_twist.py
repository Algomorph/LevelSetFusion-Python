# common libs
import numpy as np

# local
from transformation import twist_vector_to_matrix


def sdf_gradient_wrt_to_twist(live_field, voxel_x, voxel_z, twist_vector):
    # sdf_gradient_wrt_to_voxel = np.array([np.gradient(live_field)[1][voxel_z, voxel_x],
    #                                       -np.gradient(live_field)[0][voxel_z, voxel_x]])
    sdf_gradient_wrt_to_voxel = np.zeros((1, 2))

    if voxel_x - 1 < 0:
        post_sdf = live_field[voxel_x + 1, voxel_z]
        if post_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 0] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 0] = post_sdf - live_field[voxel_x, voxel_z]
    elif voxel_x + 1 > live_field.shape[0] - 1:
        pre_sdf = live_field[voxel_x - 1, voxel_z]
        if pre_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 0] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 0] = live_field[voxel_x, voxel_z] - pre_sdf
    else:
        pre_sdf = live_field[voxel_x - 1, voxel_z]
        post_sdf = live_field[voxel_x + 1, voxel_z]
        if (post_sdf < -1) or (pre_sdf < -1):
            sdf_gradient_wrt_to_voxel[0, 0] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 0] = (post_sdf - pre_sdf) / 2

    if voxel_z - 1 < 0:
        post_sdf = live_field[voxel_x, voxel_z + 1]
        if post_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 1] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 1] = post_sdf - live_field[voxel_x, voxel_z]
    elif voxel_z + 1 > live_field.shape[1] - 1:
        pre_sdf = live_field[voxel_x, voxel_z - 1]
        if pre_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 1] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 1] = live_field[voxel_x, voxel_z] - pre_sdf
    else:
        pre_sdf = live_field[voxel_x, voxel_z - 1]
        post_sdf = live_field[voxel_x, voxel_z + 1]
        if (post_sdf < -1) or (pre_sdf < -1):
            sdf_gradient_wrt_to_voxel[0, 1] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 1] = (post_sdf - pre_sdf) / 2

    twist_matrix_homo = twist_vector_to_matrix(twist_vector)
    voxel_transformed_homo = np.dot(np.linalg.inv(twist_matrix_homo), np.array([[voxel_x], [voxel_z], [1]]))
    voxel_transformed = np.delete(voxel_transformed_homo, 2)
    voxel_gradient_wrt_to_twist = np.concatenate((np.identity(2),
                                                   np.array([[-voxel_transformed[1]], [voxel_transformed[0]]])),
                                                   axis=1)

    return np.dot(sdf_gradient_wrt_to_voxel, voxel_gradient_wrt_to_twist).reshape((1, -1))
