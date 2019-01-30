#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  For transformation in 2D.
#  ================================================================
import math
import numpy as np


def twist_vector_to_matrix(twist):
    # for transforming translation and rotation vector to homo matrix in 2D

    # rotation3DMatrix = cv2.Rodrigues(rotation3DVector)

    theta = twist[2]
    twist_matrix_homo = np.identity(3)
    twist_matrix_homo[0, 0] = math.cos(theta)
    twist_matrix_homo[0, 1] = -math.sin(theta)
    twist_matrix_homo[1, 0] = math.sin(theta)
    twist_matrix_homo[1, 1] = math.cos(theta)
    twist_matrix_homo[0, 2] = twist[0]
    twist_matrix_homo[1, 2] = twist[1]

    return twist_matrix_homo  # 3 by 3 matrix


def affine_of_voxel2d(field, twist, depth_image, camera, array_offset, voxel_size,
                      camera_extrinsic_matrix=np.eye(3, dtype=np.float32),
                      narrow_band_width_voxels=1):
    twist = twist_vector_to_matrix(twist)
    projection_matrix = camera.intrinsics.intrinsic_matrix
    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units
    twisted_field = np.ones_like(field) * -999.

    y_voxel = 0.0
    w_voxel = 1.0
    for y_field in range(field.shape[1]):
        for x_field in range(field.shape[0]):
            x_voxel = (x_field + array_offset[0]) * voxel_size
            z_voxel = (y_field + array_offset[2]) * voxel_size  # acts as "Z" coordinate

            point = np.array([[x_voxel, z_voxel, w_voxel]], dtype=np.float32).T
            point = np.dot(twist, point)
            point_in_camera_space = camera_extrinsic_matrix.dot(point).flatten()

            if point_in_camera_space[1] <= 0:
                continue

            image_x_coordinate = int(
                projection_matrix[0, 0] * point_in_camera_space[0] / point_in_camera_space[1]
                + projection_matrix[0, 2] + 0.5)

            if image_x_coordinate < 0 or image_x_coordinate >= depth_image.shape[0]:
                # twisted_field[x_field, y_field] = -999.
                continue

            depth = depth_image[image_x_coordinate] * depth_ratio

            if depth <= 0.0:
                continue

            # print(depth, point_in_camera_space[1])
            signed_distance_to_voxel_along_camera_ray = depth - point_in_camera_space[1]
            # print(signed_distance_to_voxel_along_camera_ray, point_in_camera_space[1])
            if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width:
                twisted_field[x_field, y_field] = -1.0
            elif signed_distance_to_voxel_along_camera_ray > narrow_band_half_width:
                twisted_field[x_field, y_field] = 1.0
            else:
                twisted_field[x_field, y_field] = signed_distance_to_voxel_along_camera_ray / narrow_band_half_width
                # print(signed_distance_to_voxel_along_camera_ray)
    # print(twisted_field == field)
    return twisted_field
