# import unittest
from unittest import TestCase
import numpy as np
from rigid_opt.transformation import twist_vector_to_matrix


class MyTestCase(TestCase):

    def test_matrix_a01(self):
        live_field = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [0.]])
        sdf_gradient_first_term = np.gradient(live_field)
        twist_matrix_homo = twist_vector_to_matrix(twist_vector)
        sdf_gradient = np.zeros((live_field.shape[0], live_field.shape[1], 3))

        for i in range(live_field.shape[0]):
            for j in range(live_field.shape[1]):
                trans = np.dot(np.linalg.inv(twist_matrix_homo), np.array([[i], [j], [1]]))
                sdf_gradient_second_term = np.array([[1, 0, -trans[1]],
                                                     [0, 1, trans[0]]])
                sdf_gradient[i, j] = np.dot(np.array([sdf_gradient_first_term[0][i, j],
                                                      sdf_gradient_first_term[1][i, j]]),
                                            sdf_gradient_second_term)
        # print(sdf_gradient)
        expected_matrix_a = np.array([[0, 0, 0],
                                      [0, 9, 9],
                                      [0, 9, 15]])
        matrix_a = np.zeros((3, 3))
        # print(matrix_a)
        for x in range(live_field.shape[0]):
            for z in range(live_field.shape[1]):
                # print(sdf_gradient[x, z])
                # print(np.dot(sdf_gradient[x, z], sdf_gradient[x, z]))
                matrix_a += np.dot(sdf_gradient[x, z][:, None], sdf_gradient[x, z][None, :])
        self.assertTrue(np.allclose(expected_matrix_a, matrix_a))
        # self.assertTrue(np.allclose(np.linalg.inv(expected_matrix_a), np.linalg.inv(matrix_a)))

    def test_matrix_a_not_singular01(self):
        matrix_a = np.array([[1, 0, 0],
                             [0, 9, 9],
                             [0, 9, 15]])
        self.assertTrue(np.isfinite(np.linalg.cond(matrix_a)))

