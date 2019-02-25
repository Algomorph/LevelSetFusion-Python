# import unittest
from unittest import TestCase
import numpy as np
from rigid_opt.sdf_gradient_field import calculate_gradient_wrt_twist
from rigid_opt.transformation import twist_vector_to_matrix


def sdf_gradient_wrt_to_twist(live_field, y_field, x_field, twist_vector, offset, voxel_size):
    sdf_gradient_wrt_to_voxel = np.zeros((1, 2))

    if y_field - 1 < 0:
        post_sdf = live_field[y_field + 1, x_field]
        if post_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 1] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 1] = post_sdf - live_field[y_field, x_field]
    elif y_field + 1 > live_field.shape[0] - 1:
        pre_sdf = live_field[y_field - 1, x_field]
        if pre_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 1] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 1] = live_field[y_field, x_field] - pre_sdf
    else:
        pre_sdf = live_field[y_field - 1, x_field]
        post_sdf = live_field[y_field + 1, x_field]
        if (post_sdf < -1) or (pre_sdf < -1):
            sdf_gradient_wrt_to_voxel[0, 1] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 1] = (post_sdf - pre_sdf) / 2

    if x_field - 1 < 0:
        post_sdf = live_field[y_field, x_field + 1]
        if post_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 0] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 0] = post_sdf - live_field[y_field, x_field]
    elif x_field + 1 > live_field.shape[1] - 1:
        pre_sdf = live_field[y_field, x_field - 1]
        if pre_sdf < -1:
            sdf_gradient_wrt_to_voxel[0, 0] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 0] = live_field[y_field, x_field] - pre_sdf
    else:
        pre_sdf = live_field[y_field, x_field - 1]
        post_sdf = live_field[y_field, x_field + 1]
        if (post_sdf < -1) or (pre_sdf < -1):
            sdf_gradient_wrt_to_voxel[0, 0] = 0
        else:
            sdf_gradient_wrt_to_voxel[0, 0] = (post_sdf - pre_sdf) / 2

    x_voxel = (x_field + offset[0])*voxel_size
    z_voxel = (y_field + offset[2])*voxel_size

    point = np.array([[x_voxel, z_voxel, 1.]], dtype=np.float32).T
    twist_matrix_homo_inv = twist_vector_to_matrix(-twist_vector)
    trans = np.dot(twist_matrix_homo_inv, point)

    voxel_gradient_wrt_to_twist = np.array([[1, 0, trans[1]],
                                            [0, 1, -trans[0]]])

    return np.dot(sdf_gradient_wrt_to_voxel/voxel_size, voxel_gradient_wrt_to_twist).reshape((1, -1))


class MyTestCase(TestCase):
    
    def test_sdf_gradient_wrt_twist01(self):
        live_field = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [0.]])
        offset = np.array([-1, -1, 1])
        voxel_size = 1
        gradient_field = calculate_gradient_wrt_twist(live_field,
                                                   twist_vector,
                                                   array_offset=offset,
                                                   voxel_size=voxel_size)

        expected_gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                expected_gradient_field[y_field, x_field] = sdf_gradient_wrt_to_twist(live_field, y_field, x_field,
                                                                                      twist_vector, offset, voxel_size)

        self.assertTrue(np.allclose(expected_gradient_field, gradient_field))

    def test_sdf_gradient_wrt_twist02(self):
        live_field = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [0.]])

        offset = np.array([-1, -1, 1])
        voxel_size = 1
        gradient_field = calculate_gradient_wrt_twist(live_field,
                                                   twist_vector,
                                                   array_offset=offset,
                                                   voxel_size=voxel_size)

        expected_gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                expected_gradient_field[y_field, x_field] = sdf_gradient_wrt_to_twist(live_field, y_field, x_field,
                                                                                      twist_vector, offset, voxel_size)

        self.assertTrue(np.allclose(expected_gradient_field, gradient_field))

    def test_sdf_gradient_wrt_twist03(self):
        live_field = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [0.]])

        offset = np.array([-1, -1, 1])
        voxel_size = 2
        gradient_field = calculate_gradient_wrt_twist(live_field,
                                                   twist_vector,
                                                   array_offset=offset,
                                                   voxel_size=voxel_size)

        expected_gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                expected_gradient_field[y_field, x_field] = sdf_gradient_wrt_to_twist(live_field, y_field, x_field,
                                                                                      twist_vector, offset, voxel_size)

        self.assertTrue(np.allclose(expected_gradient_field, gradient_field))

    def test_sdf_gradient_wrt_twist04(self):
        live_field = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        twist_vector = np.array([[0.],
                                 [1.],
                                 [0.]])

        offset = np.array([-1, -1, 1])
        voxel_size = 1
        gradient_field = calculate_gradient_wrt_twist(live_field,
                                                   twist_vector,
                                                   array_offset=offset,
                                                   voxel_size=voxel_size)

        expected_gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                expected_gradient_field[y_field, x_field] = sdf_gradient_wrt_to_twist(live_field, y_field, x_field,
                                                                                      twist_vector, offset, voxel_size)

        self.assertTrue(np.allclose(expected_gradient_field, gradient_field))

    def test_sdf_gradient_wrt_twist05(self):
        live_field = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        twist_vector = np.array([[0.],
                                 [-1.],
                                 [0.]])

        offset = np.array([-1, -1, 1])
        voxel_size = 1
        gradient_field = calculate_gradient_wrt_twist(live_field,
                                                   twist_vector,
                                                   array_offset=offset,
                                                   voxel_size=voxel_size)

        expected_gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                expected_gradient_field[y_field, x_field] = sdf_gradient_wrt_to_twist(live_field, y_field, x_field,
                                                                                      twist_vector, offset, voxel_size)

        self.assertTrue(np.allclose(expected_gradient_field, gradient_field))

    def test_sdf_gradient_wrt_twist06(self):
        live_field = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [.5]])

        offset = np.array([-1, -1, 1])
        voxel_size = 0.5
        gradient_field = calculate_gradient_wrt_twist(live_field,
                                                   twist_vector,
                                                   array_offset=offset,
                                                   voxel_size=voxel_size)

        expected_gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                expected_gradient_field[y_field, x_field] = sdf_gradient_wrt_to_twist(live_field, y_field, x_field,
                                                                                      twist_vector, offset, voxel_size)

        self.assertTrue(np.allclose(expected_gradient_field, gradient_field))
