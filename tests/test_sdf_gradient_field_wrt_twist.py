# import unittest
from unittest import TestCase
import numpy as np
from rigid_opt.sdf_gradient_field import GradientField
from rigid_opt.transformation import twist_vector_to_matrix


class MyTestCase(TestCase):

    def test_sdf_gradient_wrt_twist01(self):
        live_field = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [0.]])
        offset = np.array([-1, -1, 1])
        twist_matrix_homo_inv = twist_vector_to_matrix(-twist_vector)
        graident_field = GradientField().calculate(live_field,
                                                   twist_vector,
                                                   array_offset=offset,
                                                   voxel_size=1)

        expected_gradient_field = np.zeros((live_field.shape[0], live_field.shape[1], 3), dtype=np.float32)

        for y_field in range(live_field.shape[0]):
            for x_field in range(live_field.shape[1]):
                x_voxel = (x_field + offset[0])
                z_voxel = (y_field + offset[2])

                point = np.array([[x_voxel, z_voxel, 1.]], dtype=np.float32).T
                trans = np.dot(twist_matrix_homo_inv, point)

                expected_gradient_field[y_field, x_field] = np.dot(np.array([np.gradient(live_field)[1][y_field,
                                                                                                        x_field],
                                                                             np.gradient(live_field)[0][y_field,
                                                                                                        x_field]]),
                                                                   np.array([[1, 0, trans[1]],
                                                                             [0, 1, -trans[0]]]))
        self.assertTrue(np.allclose(expected_gradient_field, graident_field))








    #     expected_sdf_gradient_first_term = np.gradient(live_field)
    #     expected_sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #     sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #
    #     for i in range(live_field.shape[0]):
    #         for j in range(live_field.shape[1]):
    #             expected_sdf_gradient_second_term = np.array([[1, 0, -j],
    #                                                           [0, 1, i]])
    #             expected_sdf_gradient[i, j] = np.dot(np.array([expected_sdf_gradient_first_term[0][i, j],
    #                                                      expected_sdf_gradient_first_term[1][i, j]]),
    #                                                      expected_sdf_gradient_second_term)
    #             sdf_gradient[i, j] = GradientField().sdf_gradient_wrt_twist(live_field, i, j, twist_vector)
    #     self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))
    #
    # def test_sdf_gradient_wrt_twist02(self):
    #     live_field = np.array([[1, 1, 1],
    #                            [0, 0, 0],
    #                            [-1, -1, -1]])
    #     twist_vector = np.array([[0.],
    #                              [0.],
    #                              [0.]])
    #
    #     expected_sdf_gradient_first_term = np.gradient(live_field)
    #     expected_sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #     sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #
    #     for i in range(live_field.shape[0]):
    #         for j in range(live_field.shape[1]):
    #             expected_sdf_gradient_second_term = np.array([[1, 0, -j],
    #                                                           [0, 1, i]])
    #             expected_sdf_gradient[i, j] = np.dot(np.array([expected_sdf_gradient_first_term[0][i, j],
    #                                                      expected_sdf_gradient_first_term[1][i, j]]),
    #                                            expected_sdf_gradient_second_term)
    #             sdf_gradient[i, j] = GradientField().sdf_gradient_wrt_twist(live_field, i, j, twist_vector)
    #             # print(sdf_gradient, expected_sdf_gradient)
    #
    #     self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))
    #
    # def test_sdf_gradient_wrt_twist03(self):
    #     live_field = np.array([[1, 1, 1],
    #                            [0, 0, 0],
    #                            [-1, -1, -1]])
    #     twist_vector = np.array([[1.],
    #                              [0.],
    #                              [0.]])
    #     expected_sdf_gradient_first_term = np.gradient(live_field)
    #     twist_matrix_homo = twist_vector_to_matrix(twist_vector)
    #     expected_sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #     sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #
    #     for i in range(live_field.shape[0]):
    #         for j in range(live_field.shape[1]):
    #             trans = np.dot(np.linalg.inv(twist_matrix_homo), np.array([[i], [j], [1]]))
    #             expected_sdf_gradient_second_term = np.array([[1, 0, -trans[1]],
    #                                                           [0, 1, trans[0]]])
    #             expected_sdf_gradient[i, j] = np.dot(np.array([expected_sdf_gradient_first_term[0][i, j],
    #                                                      expected_sdf_gradient_first_term[1][i, j]]),
    #                                            expected_sdf_gradient_second_term)
    #             sdf_gradient[i, j] = GradientField().sdf_gradient_wrt_twist(live_field, i, j, twist_vector)
    #             # print(sdf_gradient, expected_sdf_gradient)
    #
    #             self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))
    #
    # def test_sdf_gradient_wrt_twist04(self):
    #     live_field = np.array([[1, 0, -1],
    #                            [1, 0, -1],
    #                            [1, 0, -1]])
    #     twist_vector = np.array([[0.],
    #                              [1.],
    #                              [0.]])
    #
    #     expected_sdf_gradient_first_term = np.gradient(live_field)
    #     twist_matrix_homo = twist_vector_to_matrix(twist_vector)
    #
    #     for i in range(live_field.shape[0]):
    #         for j in range(live_field.shape[1]):
    #             trans = np.dot(np.linalg.inv(twist_matrix_homo), np.array([[i], [j], [1]]))
    #             expected_sdf_gradient_second_term = np.array([[1, 0, -trans[1]],
    #                                                           [0, 1, trans[0]]])
    #             expected_sdf_gradient = np.dot(np.array([expected_sdf_gradient_first_term[0][i, j],
    #                                                      expected_sdf_gradient_first_term[1][i, j]]),
    #                                                      expected_sdf_gradient_second_term)
    #             sdf_gradient = GradientField().sdf_gradient_wrt_twist(live_field, i, j, twist_vector)
    #             # print(sdf_gradient, expected_sdf_gradient)
    #
    #             self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))
    #
    # def test_sdf_gradient_wrt_twist05(self):
    #     live_field = np.array([[1, 0, -1],
    #                            [1, 0, -1],
    #                            [1, 0, -1]])
    #     twist_vector = np.array([[0.],
    #                              [-1.],
    #                              [0.]])
    #
    #     expected_sdf_gradient_first_term = np.gradient(live_field)
    #     twist_matrix_homo = twist_vector_to_matrix(twist_vector)
    #     expected_sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #     sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #
    #     for i in range(live_field.shape[0]):
    #         for j in range(live_field.shape[1]):
    #             trans = np.dot(np.linalg.inv(twist_matrix_homo), np.array([[i], [j], [1]]))
    #             expected_sdf_gradient_second_term = np.array([[1, 0, -trans[1]],
    #                                                           [0, 1, trans[0]]])
    #             expected_sdf_gradient[i, j] = np.dot(np.array([expected_sdf_gradient_first_term[0][i, j],
    #                                                      expected_sdf_gradient_first_term[1][i, j]]),
    #                                                      expected_sdf_gradient_second_term)
    #             sdf_gradient[i, j] = GradientField().sdf_gradient_wrt_twist(live_field, i, j, twist_vector)
    #             # print(sdf_gradient, expected_sdf_gradient)
    #
    #             self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))
    #
    # def test_sdf_gradient_wrt_twist06(self):
    #     live_field = np.array([[1, 0, -1],
    #                            [1, 0, -1],
    #                            [1, 0, -1]])
    #     twist_vector = np.array([[0.],
    #                              [0.],
    #                              [.5]])
    #
    #     expected_sdf_gradient_first_term = np.gradient(live_field)
    #     twist_matrix_homo = twist_vector_to_matrix(twist_vector)
    #     expected_sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #     sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))
    #
    #     for i in range(live_field.shape[0]):
    #         for j in range(live_field.shape[1]):
    #             trans = np.dot(np.linalg.inv(twist_matrix_homo), np.array([[i], [j], [1]]))
    #
    #             expected_sdf_gradient_second_term = np.array([[1, 0, -trans[1]],
    #                                                           [0, 1, trans[0]]])
    #             expected_sdf_gradient[i, j] = np.dot(np.array([expected_sdf_gradient_first_term[0][i, j],
    #                                                      expected_sdf_gradient_first_term[1][i, j]]),
    #                                                      expected_sdf_gradient_second_term)
    #             sdf_gradient[i, j] = GradientField().sdf_gradient_wrt_twist(live_field, i, j, twist_vector)
    #
    #             self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))
