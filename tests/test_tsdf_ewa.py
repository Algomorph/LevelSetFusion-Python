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
import tests.test_data.ewa_test_data as data
import tsdf.ewa as ewa
import os.path
import cv2

import level_set_fusion_optimization as cpp_module


class TsdfTest(TestCase):

    @staticmethod
    def image_load_helper(filename):
        path = os.path.join("tests/test_data", filename)
        if not os.path.exists(path):
            path = os.path.join("test_data", filename)
        depth_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        max_depth = np.iinfo(np.uint16).max
        depth_image[depth_image == 0] = max_depth
        return depth_image

    def test_2D_ewa_tsdf_generation1(self):
        depth_image = np.zeros((3, 640), dtype=np.uint16)
        depth_image[:] = np.iinfo(np.uint16).max
        depth_image_region = np.array([[3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326],
                                       [3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326],
                                       [3233, 3246, 3243, 3256, 3253, 3268, 3263, 3279, 3272, 3289, 3282,
                                        3299, 3291, 3308, 3301, 3317, 3310, 3326]], dtype=np.uint16)
        depth_image[:, 399:417] = depth_image_region
        camera_intrinsic_matrix = np.array([[700., 0., 320.],
                                            [0., 700., 240.],
                                            [0., 0., 1.]], dtype=np.float32)
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 3), intrinsic_matrix=camera_intrinsic_matrix),
                                 depth_unit_ratio=0.001)
        field = \
            ewa.generate_tsdf_2d_ewa_image(depth_image, camera, 1,
                                           field_size=16,
                                           array_offset=np.array([94, -256, 804]),
                                           voxel_size=0.004)
        self.assertTrue(np.allclose(field, data.out_sdf_field01, atol=2e-5))

        parameters = cpp_module.tsdf.Parameters2d()
        parameters.interpolation_method = cpp_module.tsdf.FilteringMethod.EWA_IMAGE_SPACE
        parameters.projection_matrix = camera_intrinsic_matrix
        parameters.array_offset = cpp_module.Vector2i(94, 804)
        parameters.field_shape = cpp_module.Vector2i(16, 16)

        generator = cpp_module.tsdf.Generator2d(parameters)
        field2 = generator.generate(depth_image, np.identity(4, dtype=np.float32), 1)
        self.assertTrue(np.allclose(field2, data.out_sdf_field01, atol=1e-6))

    def test_2d_ewa_tsdf_generation2(self):
        filename = "zigzag2_depth_00108.png"
        depth_image = self.image_load_helper(filename)
        test_full_image = False
        camera_intrinsic_matrix = np.array([[700., 0., 320.],
                                            [0., 700., 240.],
                                            [0., 0., 1.]], dtype=np.float32)
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 480), intrinsic_matrix=camera_intrinsic_matrix),
                                 depth_unit_ratio=0.001)

        offset_full_image = np.array([-256, 0, 0])
        chunk_x_start = 210
        chunk_y_start = 103
        chunk_size = 16
        offset_chunk_from_image = np.array([chunk_x_start, 0, chunk_y_start])
        offset_chunk = offset_full_image + offset_chunk_from_image

        if test_full_image:
            parameters = cpp_module.tsdf.Parameters2d()
            parameters.projection_matrix = camera_intrinsic_matrix
            parameters.field_shape = cpp_module.Vector2i(512, 512)
            parameters.array_offset = cpp_module.Vector2i(int(offset_full_image[0]), int(offset_full_image[2]))
            parameters.interpolation_method = cpp_module.tsdf.FilteringMethod.EWA_IMAGE_SPACE

            generator = cpp_module.tsdf.Generator2d(parameters)
            field2 = generator.generate(depth_image, np.identity(4, dtype=np.float32), 200)
            chunk = field2[chunk_y_start:chunk_y_start + chunk_size, chunk_x_start:chunk_x_start + chunk_size].copy()
        else:
            parameters = cpp_module.tsdf.Parameters2d()
            parameters.interpolation_method = cpp_module.tsdf.FilteringMethod.EWA_IMAGE_SPACE
            parameters.projection_matrix = camera_intrinsic_matrix
            parameters.array_offset = cpp_module.Vector2i(int(offset_chunk[0]), int(offset_chunk[2]))
            parameters.field_shape = cpp_module.Vector2i(16, 16)

            generator = cpp_module.tsdf.Generator2d(parameters)
            chunk = generator.generate(depth_image, np.identity(4, dtype=np.float32), 200)
        self.assertTrue(np.allclose(chunk, data.out_sdf_chunk))

        field = \
            ewa.generate_tsdf_2d_ewa_image(depth_image, camera, 200,
                                           field_size=chunk_size,
                                           array_offset=offset_chunk,
                                           voxel_size=0.004)
        self.assertTrue(np.allclose(field, data.out_sdf_chunk, atol=2e-5))

    def test_2D_ewa_tsdf_generation3(self):
        filename = "zigzag1_depth_00064.png"
        depth_image = self.image_load_helper(filename)
        camera_intrinsic_matrix = np.array([[700., 0., 320.],
                                            [0., 700., 240.],
                                            [0., 0., 1.]], dtype=np.float32)
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 3), intrinsic_matrix=camera_intrinsic_matrix),
                                 depth_unit_ratio=0.001)
        field = \
            ewa.generate_tsdf_2d_ewa_tsdf(depth_image, camera, 200,
                                          field_size=16,
                                          array_offset=np.array([-232, -256, 490]),
                                          voxel_size=0.004,
                                          gaussian_covariance_scale=0.5
                                          )

        self.assertTrue(np.allclose(field, data.out_sdf_field03))
        parameters = cpp_module.tsdf.Parameters2d()
        parameters.interpolation_method = cpp_module.tsdf.FilteringMethod.EWA_VOXEL_SPACE
        parameters.projection_matrix = camera_intrinsic_matrix
        parameters.array_offset = cpp_module.Vector2i(-232, 490)
        parameters.field_shape = cpp_module.Vector2i(16, 16)
        parameters.smoothing_factor = 0.5

        generator = cpp_module.tsdf.Generator2d(parameters)
        field2 = generator.generate(depth_image, np.identity(4, dtype=np.float32), 1)

        self.assertTrue(np.allclose(field2, data.out_sdf_field03, atol=1e-5))

    def test_2D_ewa_tsdf_generation4(self):
        filename = "zigzag1_depth_00064.png"
        depth_image = self.image_load_helper(filename)
        camera_intrinsic_matrix = np.array([[700., 0., 320.],
                                            [0., 700., 240.],
                                            [0., 0., 1.]], dtype=np.float32)
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 3), intrinsic_matrix=camera_intrinsic_matrix),
                                 depth_unit_ratio=0.001)
        field = \
            ewa.generate_tsdf_2d_ewa_tsdf_inclusive(depth_image, camera, 200,
                                                    field_size=16,
                                                    array_offset=np.array([-232, -256, 490]),
                                                    voxel_size=0.004,
                                                    gaussian_covariance_scale=0.5
                                                    )

        self.assertTrue(np.allclose(field, data.out_sdf_field04))

        parameters = cpp_module.tsdf.Parameters2d()
        parameters.interpolation_method = cpp_module.tsdf.FilteringMethod.EWA_VOXEL_SPACE_INCLUSIVE
        parameters.projection_matrix = camera_intrinsic_matrix
        parameters.array_offset = cpp_module.Vector2i(-232, 490)
        parameters.field_shape = cpp_module.Vector2i(16, 16)
        parameters.smoothing_factor = 0.5

        generator = cpp_module.tsdf.Generator2d(parameters)
        field2 = generator.generate(depth_image, np.identity(4, dtype=np.float32), 1)
        self.assertTrue(np.allclose(field2, data.out_sdf_field04, atol=1e-5))

    def test_3d_ewa_tsdf_generation1(self):
        filename = "zigzag2_depth_00108.png"
        depth_image = self.image_load_helper(filename)
        array_offset = np.array([-46, -8, 105], dtype=np.int32)
        field_shape = np.array([16, 1, 16], dtype=np.int32)
        camera_intrinsic_matrix = np.array([[700., 0., 320.],
                                            [0., 700., 240.],
                                            [0., 0., 1.]], dtype=np.float32)
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 480), intrinsic_matrix=camera_intrinsic_matrix),
                                 depth_unit_ratio=0.001)

        parameters = cpp_module.tsdf.Parameters3d()
        parameters.interpolation_method = cpp_module.tsdf.FilteringMethod.EWA_IMAGE_SPACE
        parameters.projection_matrix = camera_intrinsic_matrix
        parameters.array_offset = cpp_module.Vector3i(-46, -8, 105)
        parameters.field_shape = cpp_module.Vector3i(16, 1, 16)
        parameters.smoothing_factor = 1.0

        generator = cpp_module.tsdf.Generator3d(parameters)
        field2 = generator.generate(depth_image, np.identity(4, dtype=np.float32), 1)
        self.assertTrue(np.allclose(field2, data.sdf_3d_slice01))

        field = \
            ewa.generate_tsdf_3d_ewa_image(depth_image, camera,
                                           field_shape=field_shape,
                                           array_offset=array_offset,
                                           voxel_size=0.004)

        self.assertTrue(np.allclose(field, data.sdf_3d_slice01, atol=1.5e-5))

    def test_3d_ewa_tsdf_generation2(self):
        filename = "zigzag2_depth_00108.png"
        depth_image = self.image_load_helper(filename)
        array_offset = np.array([-46, -8, 105], dtype=np.int32)
        field_shape = np.array([16, 1, 16], dtype=np.int32)
        camera_intrinsic_matrix = np.array([[700., 0., 320.],
                                            [0., 700., 240.],
                                            [0., 0., 1.]], dtype=np.float32)
        camera = cam.DepthCamera(intrinsics=cam.Camera.Intrinsics((640, 480), intrinsic_matrix=camera_intrinsic_matrix),
                                 depth_unit_ratio=0.001)

        parameters = cpp_module.tsdf.Parameters3d()
        parameters.interpolation_method = cpp_module.tsdf.FilteringMethod.EWA_IMAGE_SPACE
        parameters.projection_matrix = camera_intrinsic_matrix
        parameters.array_offset = cpp_module.Vector3i(-46, -8, 105)
        parameters.field_shape = cpp_module.Vector3i(16, 1, 16)
        parameters.smoothing_factor = 0.5

        generator = cpp_module.tsdf.Generator3d(parameters)
        field2 = generator.generate(depth_image, np.identity(4, dtype=np.float32), 1)

        self.assertTrue(np.allclose(field2, data.sdf_3d_slice02, atol=1e-5))

        field = \
            ewa.generate_tsdf_3d_ewa_image(depth_image, camera,
                                           field_shape=field_shape,
                                           array_offset=array_offset,
                                           voxel_size=0.004,
                                           gaussian_covariance_scale=0.5)

        self.assertTrue(np.allclose(field, data.sdf_3d_slice02))
