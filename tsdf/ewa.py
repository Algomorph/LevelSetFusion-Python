#  ================================================================
#  Created by Gregory Kramida on 1/21/19.
#  Copyright (c) 2019 Gregory Kramida
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

import numpy as np

# TODO: WIP

def generate_3d_tsdf_field_from_depth_image_ewa(depth_image, camera,
                                                camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                field_size=128, default_value=1, voxel_size=0.004,
                                                array_offset=np.array([-64, -64, 64]),
                                                narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    """
    Assumes camera is at array_offset voxels relative to sdf grid
    :param narrow_band_width_voxels:
    :param array_offset:
    :param camera_extrinsic_matrix: matrix representing transformation of the camera (incl. rotation and translation)
    [ R | T]
    [ 0 | 1]
    :param voxel_size: voxel size, in meters
    :param default_value: default initial TSDF value
    :param field_size:
    :param depth_image:
    :type depth_image: np.ndarray
    :param camera:
    :type camera: calib.camera.DepthCamera
    :param image_y_coordinate:
    :type image_y_coordinate: int
    :return:
    """
    # TODO: use back_cutoff_voxels for additional limit on
    # "if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width" (maybe?)

    if default_value == 1:
        field = np.ones((field_size, field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    projection_matrix = camera.intrinsics.intrinsic_matrix

    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    w_voxel = 1.0

    camera_rotation_matrix = camera_extrinsic_matrix[0:3, 0:3]
    # TODO: we don't **actually** need to compute these, do we now?
    #  We just need J*W, right?
    #  Cause that would correspond to M in Q = MM^-1, right?
    # covariance_voxel_sphere_world_space = np.eye(3)  # TODO: * voxel_size? -- YES!!!
    # covariance_camera_space = camera_rotation_matrix.dot(covariance_voxel_sphere_world_space) \
    #     .dot(camera_rotation_matrix.T)

    radius_threshold = 4.0

    for z_field in range(field_size):
        for y_field in range(field_size):
            for x_field in range(field_size):

                x_voxel = (x_field + array_offset[0]) * voxel_size
                y_voxel = (y_field + array_offset[1]) * voxel_size
                z_voxel = (z_field + array_offset[2]) * voxel_size

                voxel_center = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
                vc = voxel_center_in_camera_space = camera_extrinsic_matrix.dot(voxel_center).flatten()[:3]

                if voxel_center_in_camera_space[2] <= 0:
                    continue

                l = ray_distance_camera_to_voxel_center = np.linalg.norm(voxel_center_in_camera_space)
                zc2 = distance_camera_to_voxel_center_sqared = voxel_center_in_camera_space[2] ** 2

                projection_jacobian = \
                    np.array([[1 / vc[2], 0, -vc[0] / zc2],
                              [0, 1 / vc[2], -vc[1] / zc2],
                              [vc[0] / l, vc[1] / l, vc[2] / l]])

                # remapped_covariance = projection_jacobian.dot(covariance_camera_space) \
                #     .dot(projection_jacobian.T)
                image_coordinates = projection_matrix.dot(voxel_center_in_camera_space) / vc[2]

                # covariance_image_space = remapped_covariance[0:3, 0:3] + np.eye(2)
                covariance_transform = projection_jacobian.dot(camera_rotation_matrix)

                # T = covariance_transform, V = T.I.T^-1 + I
                u_x = covariance_transform[0, 0]
                v_x = covariance_transform[0, 1]
                u_y = covariance_transform[1, 0]
                v_y = covariance_transform[1, 1]

                det_Rinv = np.linalg.det(camera_rotation_matrix.T)
                det_Jinv = np.linalg.det(np.linalg.inv(projection_jacobian))
                normalization_factor = 1 / (det_Jinv * det_Rinv)

                A = v_x * v_x + v_y * v_y + 1
                B = -2 * (u_x * v_x + u_y + v_y)
                C = u_x * u_x + u_y * u_y + 1
                F = A * C - B * B  # / 4.0
                # TODO: is this correct? Do we need to scale up f to ... ? and a,b,c accordingly
                factor = 1.0 / F # 4 / F if F was scaled down by 4
                A *= factor
                B *= factor




                # ax1 = covariance_transform[0, :] * c  # do we need to multiply by f here?? Or by c?
                # ax2 = covariance_transform[1, :] * c
                #
                # ellipse_bbox = np.array([ax1 + ax2, ax1 - ax2, -ax1 + ax2, -ax1 - ax2])
                #
                # x_min = int(ellipse_bbox[:, 0].min() + 0.5)
                # x_max = int(ellipse_bbox[:, 0].max())
                # y_min = int(ellipse_bbox[:, 1].min() + 0.5)
                # y_max = int(ellipse_bbox[:, 1].max())

                depth_sample_x_coordinate = int(image_coordinates[0] + 0.5)
                depth_sample_y_coordinate = int(image_coordinates[1] + 0.5)

                if depth_sample_x_coordinate < 0 or depth_sample_x_coordinate >= depth_image.shape[1] \
                        or depth_sample_y_coordinate < 0 or depth_sample_y_coordinate >= depth_image.shape[0]:
                    continue

                surface_depth = depth_image[depth_sample_y_coordinate, depth_sample_x_coordinate] * depth_ratio
                point_on_surface = voxel_center_in_camera_space * surface_depth / voxel_center_in_camera_space[2]

                if surface_depth <= 0.0:
                    continue

                signed_distance_to_voxel_along_camera_ray = surface_depth - voxel_center_in_camera_space[2]

                if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width:
                    field[y_field, x_field] = -1.0
                elif signed_distance_to_voxel_along_camera_ray > narrow_band_half_width:
                    field[y_field, x_field] = 1.0
                else:
                    field[y_field, x_field] = signed_distance_to_voxel_along_camera_ray / narrow_band_half_width

    return field


def generate_2d_tsdf_field_from_depth_image_ewa(depth_image, camera, image_y_coordinate,
                                                camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                field_size=128, default_value=1, voxel_size=0.004,
                                                array_offset=np.array([-64, -64, 64]),
                                                narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    """
    Assumes camera is at array_offset voxels relative to sdf grid
    :param narrow_band_width_voxels:
    :param array_offset:
    :param camera_extrinsic_matrix: matrix representing transformation of the camera (incl. rotation and translation)
    [ R | T]
    [ 0 | 1]
    :param voxel_size: voxel size, in meters
    :param default_value: default initial TSDF value
    :param field_size:
    :param depth_image:
    :type depth_image: np.ndarray
    :param camera:
    :type camera: calib.camera.DepthCamera
    :param image_y_coordinate:
    :type image_y_coordinate: int
    :return:
    """
    # TODO: use back_cutoff_voxels for additional limit on
    # "if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width" (maybe?)

    if default_value == 1:
        field = np.ones((field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    projection_matrix = camera.intrinsics.intrinsic_matrix
    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    y_voxel = 0.0
    w_voxel = 1.0

    for y_field in range(field_size):
        for x_field in range(field_size):
            x_voxel = (x_field + array_offset[0]) * voxel_size
            z_voxel = (y_field + array_offset[2]) * voxel_size  # acts as "Z" coordinate

            point = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
            point_in_camera_space = camera_extrinsic_matrix.dot(point).flatten()

            if point_in_camera_space[2] <= 0:
                continue

            image_x_coordinate = int(
                projection_matrix[0, 0] * point_in_camera_space[0] / point_in_camera_space[2]
                + projection_matrix[0, 2] + 0.5)

            if image_x_coordinate < 0 or image_x_coordinate >= depth_image.shape[1]:
                continue

            depth = depth_image[image_y_coordinate, image_x_coordinate] * depth_ratio

            if depth <= 0.0:
                continue

            signed_distance_to_voxel_along_camera_ray = depth - point_in_camera_space[2]

            if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width:
                field[y_field, x_field] = -1.0
            elif signed_distance_to_voxel_along_camera_ray > narrow_band_half_width:
                field[y_field, x_field] = 1.0
            else:
                field[y_field, x_field] = signed_distance_to_voxel_along_camera_ray / narrow_band_half_width

    return field
