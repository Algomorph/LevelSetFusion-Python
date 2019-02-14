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
import math_utils.elliptical_gaussians as eg

# C++ extension
import level_set_fusion_optimization as cpp_extension


def generate_3d_tsdf_field_from_depth_image_ewa(depth_image, camera,
                                                camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                field_size=128, default_value=1, voxel_size=0.004,
                                                array_offset=np.array([-64, -64, 64]),
                                                narrow_band_width_voxels=20, back_cutoff_voxels=np.inf,
                                                visualize_samples=True):
    """
    Assumes camera is at array_offset voxels relative to sdf grid
    :param narrow_band_width_voxels: span (in voxels) where signed distance is between -1 and 1
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
    # "if signed_distance < -narrow_band_half_width" (maybe?)

    if default_value == 1:
        field = np.ones((field_size, field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    camera_intrinsic_matrix = camera.intrinsics.intrinsic_matrix
    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    w_voxel = 1.0

    camera_rotation_matrix = camera_extrinsic_matrix[0:3, 0:3]
    covariance_voxel_sphere_world_space = np.eye(3) * voxel_size
    covariance_camera_space = camera_rotation_matrix.dot(covariance_voxel_sphere_world_space) \
        .dot(camera_rotation_matrix.T)

    squared_radius_threshold = 4.0

    for x_field in range(field_size):
        for y_field in range(field_size):
            for z_field in range(field_size):

                x_voxel = (x_field + array_offset[0]) * voxel_size
                y_voxel = (y_field + array_offset[1]) * voxel_size
                z_voxel = (z_field + array_offset[2]) * voxel_size

                voxel_world = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
                voxel_camera = camera_extrinsic_matrix.dot(voxel_world).flatten()[:3]

                if voxel_camera[2] <= 0:
                    continue

                # distance along ray from camera to voxel center
                ray_distance = np.linalg.norm(voxel_camera)
                # squared distance along optical axis from camera to voxel
                z_cam_squared = voxel_camera[2] ** 2
                inv_z_cam = 1 / voxel_camera[2]

                projection_jacobian = \
                    np.array([[inv_z_cam, 0, -voxel_camera[0] / z_cam_squared],
                              [0, inv_z_cam, -voxel_camera[1] / z_cam_squared],
                              [voxel_camera[0] / ray_distance, voxel_camera[1] / ray_distance,
                               voxel_camera[2] / ray_distance]])

                remapped_covariance = projection_jacobian.dot(covariance_camera_space) \
                    .dot(projection_jacobian.T)

                # TODO: inv troubles?
                Q = np.linalg.inv(remapped_covariance[0:3, 0:3]) + np.eye(2)
                # Q = remapped_covariance[0:3, 0:3] + np.eye(2)
                gaussian = eg.EllipticalGaussian(eg.ImplicitEllipse(Q=Q, F=squared_radius_threshold))

                voxel_image = (camera_intrinsic_matrix.dot(voxel_camera) / voxel_camera[2])[:2]
                voxel_image = voxel_image.reshape(-1, 1)

                bounds = gaussian.ellipse.get_bounds() + voxel_image
                (start_x, start_y) = (bounds[:, 0] + 0.5).astype(np.int32)
                (end_x, end_y) = (bounds[:, 1] + 1.5).astype(np.int32)

                if end_y <= 0 or start_y > depth_image.shape[0] or end_x <= 0 or start_x > depth_image.shape[1]:
                    continue
                start_y = max(0, start_y)
                end_y = min(depth_image.shape[0], end_y)
                start_x = max(0, start_x)
                end_x = min(depth_image.shape[1], end_x)

                weights_sum = 0.0
                depth_sum = 0

                for y_sample in range(start_y, end_y):
                    for x_sample in range(start_x, end_x):
                        sample_centered = np.array([[x_sample],
                                                    [y_sample]], dtype=np.float64) - voxel_image

                        dist_sq = gaussian.get_distance_from_center_squared(sample_centered)
                        if dist_sq > squared_radius_threshold:
                            continue
                        weight = gaussian.compute(dist_sq)

                        surface_depth = depth_image[y_sample, x_sample] * depth_ratio
                        if surface_depth <= 0.0:
                            continue
                        depth_sum += weight * surface_depth
                        weights_sum += weight

                final_depth = depth_sum / weights_sum
                if depth_sum <= 0.0:
                    continue

                signed_distance = final_depth - voxel_camera[2]

                if signed_distance < -narrow_band_half_width:
                    field[x_field, y_field, z_field] = -1.0
                elif signed_distance > narrow_band_half_width:
                    field[x_field, y_field, z_field] = 1.0
                else:
                    field[x_field, y_field, z_field] = signed_distance / narrow_band_half_width

    return field


def generate_2d_tsdf_field_from_depth_image_ewa_cpp(depth_image, camera, image_y_coordinate,
                                                    camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                    field_size=128, default_value=1, voxel_size=0.004,
                                                    array_offset=np.array([-64, -64, 64], dtype=np.int32),
                                                    narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    if type(array_offset) != np.ndarray:
        array_offset = np.array(array_offset).astype(np.int32)
    return cpp_extension.generate_2d_tsdf_field_from_depth_image_ewa(image_y_coordinate,
                                                                     depth_image,
                                                                     camera.depth_unit_ratio,
                                                                     camera.intrinsics.intrinsic_matrix.astype(
                                                                         np.float32),
                                                                     camera_extrinsic_matrix.astype(np.float32),
                                                                     array_offset.astype(np.int32),
                                                                     field_size,
                                                                     voxel_size,
                                                                     narrow_band_width_voxels)


def generate_3d_tsdf_field_from_depth_image_ewa_cpp(depth_image, camera,
                                                    camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                    field_shape=np.array([128, 128, 128], dtype=np.int32),
                                                    default_value=1, voxel_size=0.004,
                                                    array_offset=np.array([-64, -64, 64], dtype=np.int32),
                                                    narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    if type(field_shape) != np.ndarray:
        field_shape = np.array(field_shape).astype(np.int32)
    if type(array_offset) != np.ndarray:
        array_offset = np.array(array_offset).astype(np.int32)
    return cpp_extension.generate_3d_tsdf_field_from_depth_image_ewa(depth_image,
                                                                     camera.depth_unit_ratio,
                                                                     camera.intrinsics.intrinsic_matrix.astype(
                                                                         np.float32),
                                                                     camera_extrinsic_matrix.astype(np.float32),
                                                                     array_offset,
                                                                     field_shape,
                                                                     voxel_size,
                                                                     narrow_band_width_voxels)


def generate_3d_tsdf_ewa_cpp_viz(depth_image, camera, field,
                                 camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                 voxel_size=0.004,
                                 array_offset=np.array([-64, -64, 64], dtype=np.int32), scale=20):
    if type(array_offset) != np.ndarray:
        array_offset = np.array(array_offset).astype(np.int32)
    return cpp_extension.generate_3d_tsdf_field_from_depth_image_ewa_viz(depth_image,
                                                                         camera.depth_unit_ratio,
                                                                         field,
                                                                         camera.intrinsics.intrinsic_matrix.astype(
                                                                             np.float32),
                                                                         camera_extrinsic_matrix.astype(np.float32),
                                                                         array_offset,
                                                                         voxel_size,
                                                                         scale)


def generate_2d_tsdf_field_from_depth_image_ewa(depth_image, camera, image_y_coordinate,
                                                camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                field_size=128, default_value=1, voxel_size=0.004,
                                                array_offset=np.array([-64, -64, 64]),
                                                narrow_band_width_voxels=20, back_cutoff_voxels=np.inf,
                                                visualize_samples=False):
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

    if default_value == 1:
        field = np.ones((field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    camera_intrinsic_matrix = camera.intrinsics.intrinsic_matrix
    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    w_voxel = 1.0
    y_voxel = 0

    camera_rotation_matrix = camera_extrinsic_matrix[0:3, 0:3]
    covariance_voxel_sphere_world_space = np.eye(3) * voxel_size
    covariance_camera_space = camera_rotation_matrix.dot(covariance_voxel_sphere_world_space) \
        .dot(camera_rotation_matrix.T)

    squared_radius_threshold = 4.0

    for y_field in range(field_size):
        for x_field in range(field_size):
            x_voxel = (x_field + array_offset[0]) * voxel_size
            z_voxel = (y_field + array_offset[2]) * voxel_size
            voxel_world = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
            voxel_camera = camera_extrinsic_matrix.dot(voxel_world).flatten()[:3]

            if voxel_camera[2] <= 0:
                continue

            # distance along ray from camera to voxel
            ray_distance = np.linalg.norm(voxel_camera)
            # squared distance along optical axis from camera to voxel
            z_cam_squared = voxel_camera[2] ** 2

            projection_jacobian = \
                np.array([[1 / voxel_camera[2], 0, -voxel_camera[0] / z_cam_squared],
                          [0, 1 / voxel_camera[2], -voxel_camera[1] / z_cam_squared],
                          [voxel_camera[0] / ray_distance, voxel_camera[1] / ray_distance,
                           voxel_camera[2] / ray_distance]])

            remapped_covariance = projection_jacobian.dot(covariance_camera_space) \
                .dot(projection_jacobian.T)

            # TODO: why is inverse not working? inverse covariance image space
            # Q = np.linalg.inv(remapped_covariance[0:2, 0:2]) + np.eye(2)
            Q = remapped_covariance[0:2, 0:2] + np.eye(2)
            gaussian = eg.EllipticalGaussian(eg.ImplicitEllipse(Q=Q, F=squared_radius_threshold))

            voxel_image = (camera_intrinsic_matrix.dot(voxel_camera) / voxel_camera[2])[:2]
            voxel_image[1] = image_y_coordinate
            voxel_image = voxel_image.reshape(-1, 1)

            bounds = gaussian.ellipse.get_bounds() + voxel_image
            (start_x, start_y) = (bounds[:, 0] + 0.5).astype(np.int32)
            (end_x, end_y) = (bounds[:, 1] + 1.5).astype(np.int32)

            if end_y <= 0 or start_y > depth_image.shape[0] or end_x <= 0 or start_x > depth_image.shape[1]:
                continue
            start_y = max(0, start_y)
            end_y = min(depth_image.shape[0], end_y)
            start_x = max(0, start_x)
            end_x = min(depth_image.shape[1], end_x)

            weights_sum = 0.0
            depth_sum = 0.0

            for y_sample in range(start_y, end_y):
                for x_sample in range(start_x, end_x):
                    sample_centered = np.array([[x_sample],
                                                [y_sample]], dtype=np.float64) - voxel_image

                    dist_sq = gaussian.get_distance_from_center_squared(sample_centered)
                    if dist_sq > squared_radius_threshold:
                        continue
                    weight = gaussian.compute(dist_sq)

                    surface_depth = depth_image[y_sample, x_sample] * depth_ratio
                    if surface_depth <= 0.0:
                        continue

                    depth_sum += weight * surface_depth
                    weights_sum += weight

            if depth_sum <= 0.0:
                continue
            final_depth = depth_sum / weights_sum

            # signed distance from surface to voxel along camera axis
            signed_distance = final_depth - voxel_camera[2]

            if signed_distance < -narrow_band_half_width:
                field[y_field, x_field] = -1.0
            elif signed_distance > narrow_band_half_width:
                field[y_field, x_field] = 1.0
            else:
                field[y_field, x_field] = signed_distance / narrow_band_half_width

    return field
