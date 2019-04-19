#  ================================================================
#  Created by Gregory Kramida on 4/19/19.
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
from enum import Enum
import tsdf.common as common


class InterpolationMethod:
    NONE = 0
    BILINEAR_IMAGE_SPACE = 1
    BILINEAR_TSDF_SPACE = 2
    EWA_IMAGE_SPACE = 3
    EWA_TSDF_IMAGE_SPACE = 5
    EWA_TSDF_TSDF_SPACE = 6
    EWA_TSDF_SPACE_INCLUSIVE = 7


class Generator:
    class GenerationParameters:

        def __init__(self, depth_unit_ratio=0.001,
                     projection_matrix=np.eye(3, 3),
                     field_size=(128, 128, 128),
                     voxel_size=0.004,
                     field_offset=np.array([-64, -64, 64]),
                     narrow_band_width_voxels=20, back_cutoff_voxels=np.inf,
                     interpolation_method=InterpolationMethod.NONE,
                     smoothing_coefficient=1.0):
            """
            Assumes camera is at array_offset voxels relative to TSDF grid
            :param narrow_band_width_voxels:
            :param field_offset: offset of the resulting field relative to world center
            :param voxel_size: voxel size, in meters
            :type field_size: numpy.ndarray
            :param field_size: size
            :return:
            """
            self.depth_unit_ratio = depth_unit_ratio
            self.projection_matrix = projection_matrix
            self.field_size = field_size
            self.voxel_size = voxel_size
            self.scene_offset = field_offset
            self.narrow_band_width_voxels = narrow_band_width_voxels
            self.back_cutoff_voxels = back_cutoff_voxels
            self.interpolation_method = interpolation_method
            self.smoothing_coefficient = smoothing_coefficient

    def __init__(self, parameters):
        self.parameters = parameters

    def generate(self, depth_image, camera_pose=np.eye(4, dtype=np.float32)):
        """
        generate a TSDF grid
        :param depth_image:
        :param camera_pose:
        :return:
        """
        field = np.ones(self.parameters.field_size, dtype=np.float32)

        projection_matrix = self.parameters.projection_matrix
        depth_ratio = self.parameters.depth_unit_ratio
        field_size = self.parameters.field_size
        array_offset = self.parameters.array_offset
        voxel_size = self.parameters.voxel_size
        # in metric units
        narrow_band_half_width = self.parameters.narrow_band_width_voxels / 2 * self.parameters.voxel_size

        w_voxel = 1.0

        for z_field in range(field_size[2]):
            for y_field in range(field_size[1]):
                for x_field in range(field_size[0]):
                    # coordinates deliberately flipped here to maintain consistency between Python & C++ implementations
                    # Eigen Tensors being used are column-major, whereas here we use row-major layout by default
                    # NB: in the future we might switch to row-major Eigen Tensors
                    # (when they become fully implemented and supported by Eigen)
                    z_voxel = (x_field + array_offset[0]) * voxel_size
                    y_voxel = (y_field + array_offset[1]) * voxel_size
                    x_voxel = (z_field + array_offset[2]) * voxel_size

                    point = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
                    point_in_camera_space = camera_pose.dot(point).flatten()

                    if point_in_camera_space[2] <= 0:
                        continue

                    image_x_coordinate = int(
                        projection_matrix[0, 0] * point_in_camera_space[0] / point_in_camera_space[2]
                        + projection_matrix[0, 2] + 0.5)
                    image_y_coordinate = int(
                        projection_matrix[1, 1] * point_in_camera_space[1] / point_in_camera_space[2]
                        + projection_matrix[1, 2] + 0.5
                    )

                    if image_x_coordinate < 0 or image_x_coordinate >= depth_image.shape[1] \
                            or image_y_coordinate < 0 or image_y_coordinate >= depth_image.shape[0]:
                        continue

                    depth = depth_image[image_y_coordinate, image_x_coordinate] * depth_ratio

                    if depth <= 0.0:
                        continue

                    signed_distance_to_voxel_along_camera_ray = depth - point_in_camera_space[2]

                    field[z_field, y_field, x_field] = common.compute_tsdf_value(
                        signed_distance_to_voxel_along_camera_ray, narrow_band_half_width)


class Generator2d(Generator):

    def __init__(self, ):
        pass
