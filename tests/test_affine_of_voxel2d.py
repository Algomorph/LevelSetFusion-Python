# import unittest
from unittest import TestCase
import numpy as np
import math
from rigid_opt.sdf_gradient_field import GradientField
from rigid_opt.transformation import twist_vector_to_matrix, affine_of_voxel2d
from calib.camera import Camera, DepthCamera

class MyTestCase(TestCase):

    def test_affine_of_voxel2d01(self):
        field = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
        twist = np.array([[0.],
                          [0.],
                          [0.]])
        depth_image = np.array([1, 1, 1])
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                               intrinsic_matrix=intrinsic_matrix))
        array_offset = np.array([-1, -1, 1])
        voxel_size = 0.004
        # camera_extrinsic_matrix = np.eye(3, dtype=np.float32),
        # narrow_band_width_voxels = 20
        expected_twisted_field = np.copy(field)
        twisted_field = affine_of_voxel2d(field, twist, depth_image, camera, array_offset, voxel_size,
                                          camera_extrinsic_matrix=np.eye(3, dtype=np.float32),
                                          narrow_band_width_voxels=1)
        print(twisted_field)
        self.assertTrue(np.allclose(twisted_field, expected_twisted_field))

    def test_affine_of_voxel2d02(self):
        field = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
        twist = np.array([[0.],
                          [0.],
                          [2 * math.pi]])
        depth_image = np.array([0, 0, 0])
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                               intrinsic_matrix=intrinsic_matrix))
        array_offset = np.array([-1, -1, 1])
        voxel_size = 0.004
        # camera_extrinsic_matrix = np.eye(3, dtype=np.float32),
        # narrow_band_width_voxels = 20
        expected_twisted_field = np.copy(field)
        twisted_field = affine_of_voxel2d(field, twist, depth_image, camera, array_offset, voxel_size,
                                          camera_extrinsic_matrix=np.eye(3, dtype=np.float32),
                                          narrow_band_width_voxels=1)
        self.assertTrue(np.allclose(twisted_field, expected_twisted_field))

    def test_affine_of_voxel2d03(self):
        field = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
        twist = np.array([[0.],
                          [0.],
                          [math.pi]])
        depth_image = np.array([0, 0, 0])
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                               intrinsic_matrix=intrinsic_matrix))
        array_offset = np.array([-1, -1, 1])
        voxel_size = 0.004
        # camera_extrinsic_matrix = np.eye(3, dtype=np.float32),
        # narrow_band_width_voxels = 20
        expected_twisted_field = np.array([[0, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, 0]])
        twisted_field = affine_of_voxel2d(field, twist, depth_image, camera, array_offset, voxel_size,
                                          camera_extrinsic_matrix=np.eye(3, dtype=np.float32),
                                          narrow_band_width_voxels=1)
        self.assertTrue(np.allclose(twisted_field, expected_twisted_field))

    def test_affine_of_voxel2d04(self):
        field = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
        twist = np.array([[0.],
                          [0.],
                          [math.pi/2.]])
        depth_image = np.array([0, 0, 0])
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                               intrinsic_matrix=intrinsic_matrix))
        array_offset = np.array([-1, -1, 1])
        voxel_size = 0.004
        # camera_extrinsic_matrix = np.eye(3, dtype=np.float32),
        # narrow_band_width_voxels = 20
        expected_twisted_field = np.array([[0, 0, 0],
                                           [0, 0, 0],
                                           [0, 0, 0]])
        twisted_field = affine_of_voxel2d(field, twist, depth_image, camera, array_offset, voxel_size,
                                          camera_extrinsic_matrix=np.eye(3, dtype=np.float32),
                                          narrow_band_width_voxels=1)
        self.assertTrue(np.allclose(twisted_field, expected_twisted_field))

    def test_affine_of_voxel2d05(self):
        field = np.array([[1, 0, -1],
                          [1, 0, -1],
                          [1, 0, -1]])
        twist = np.array([[0.],
                          [.004],
                          [0.]])
        depth_image = np.array([0, 0, 0])
        intrinsic_matrix = np.array([[1, 0, 1],  # FX = 1 CX = 1
                                     [0, 1, 1],  # FY = 1 CY = 1
                                     [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(3, 3),
                                                               intrinsic_matrix=intrinsic_matrix))
        array_offset = np.array([-1, -1, 1])
        voxel_size = 0.004
        # camera_extrinsic_matrix = np.eye(3, dtype=np.float32),
        # narrow_band_width_voxels = 20
        expected_twisted_field = np.array([[1, 1, 0],
                                           [1, 1, 0],
                                           [1, 1, 0]])
        twisted_field = affine_of_voxel2d(field, twist, depth_image, camera, array_offset, voxel_size,
                                          camera_extrinsic_matrix=np.eye(3, dtype=np.float32),
                                          narrow_band_width_voxels=1)
        self.assertTrue(np.allclose(twisted_field, expected_twisted_field))
