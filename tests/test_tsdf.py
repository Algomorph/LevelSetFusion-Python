#  ================================================================
#  Created by Gregory Kramida on 1/31/19.
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

from unittest import TestCase
import numpy as np
import calib.camera as cam
import tests.ewa_test_data as data
import tsdf.ewa as ewa


class TsdfTest(TestCase):
    def test_ewa_tsdf_generation(self):
        depth_image = np.zeros((3, 640), dtype=np.uint16)
        depth_image[:] = np.iinfo(np.uint16).max
        depth_image_region = np.array([[3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326],
                                       [3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326],
                                       [3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326]], dtype=np.uint16)
        depth_image[:, 399:417] = depth_image_region
        camera_intrisic_matrix = np.array([[700., 0., 320.],
                                           [0., 700., 240.],
                                           [0., 0., 1.]])
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 3), intrinsic_matrix=camera_intrisic_matrix),
                                 depth_unit_ratio=0.001)
        field = \
            ewa.generate_2d_tsdf_field_from_depth_image_ewa(depth_image, camera, 1,
                                                            field_size=16,
                                                            array_offset=np.array([94, -256, 804]),
                                                            voxel_size=0.004)
        print(repr(field))
        self.assertTrue(np.allclose(field, data.out_sdf_field))
