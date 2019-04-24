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
import level_set_fusion_optimization as cpp
from tsdf.generation import generate_2d_tsdf_field_from_depth_image
from calib.camera import DepthCamera, Camera


class Generator2d:
    def __init__(self, parameters):
        self.parameters = parameters
        self.camera = DepthCamera(
            # dummy resolution
            intrinsics=Camera.Intrinsics(resolution=(480, 640), intrinsic_matrix=parameters.projection_matrix),
            depth_unit_ratio=parameters.depth_unit_ratio)

    def generate(self, depth_image, camera_pose=np.eye(4, dtype=np.float32), image_y_coordinate=0):
        """
        Generate a 2D TSDF grid from the specified row of the depth image assuming the given camera pose.
        :param depth_image: image composed of depth values
        :param camera_pose: camera pose relative to world
        :param image_y_coordinate: y coordinate corresponding to the row to use in the depth image
        :return: a tsdf grid
        """
        generate_2d_tsdf_field_from_depth_image(depth_image, self.camera, image_y_coordinate,
                                                camera_pose, self.parameters.field_shape[0], 1.0,
                                                self.parameters.voxel_size, self.parameters.array_offset,
                                                self.parameters.narrow_band_width_voxels,
                                                interpolation_method=self.parameters.interpolation_method,
                                                smoothing_coefficient=self.parameters.smoothing_factor)
