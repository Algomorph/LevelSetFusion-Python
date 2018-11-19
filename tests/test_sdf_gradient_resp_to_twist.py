# import unittest
from unittest import TestCase
import numpy as np
from sdf_gradient_resp_to_twist import sdf_gradient_resp_to_twist


class MyTestCase(TestCase):

    def test_sdf_gradient_resp_to_twist01(self):
        live_field = np.array([[1, 0, -1],
                               [1, 0, -1],
                               [1, 0, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [0.]])

        expected_sdf_gradient_first_term = np.gradient(live_field)

        for i in range(live_field.shape[0]):
            for j in range(live_field.shape[1]):
                expected_sdf_gradient_second_term = np.array([[1, 0, -i],
                                                              [0, 1, j]])
                expected_sdf_gradient = np.dot(np.array([expected_sdf_gradient_first_term[1][i, j],
                                                         -expected_sdf_gradient_first_term[0][i, j]]),
                                               expected_sdf_gradient_second_term)
                sdf_gradient = sdf_gradient_resp_to_twist(live_field, j, i, twist_vector)

                self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))

    def test_sdf_gradient_resp_to_twist02(self):
        live_field = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])
        twist_vector = np.array([[0.],
                                 [0.],
                                 [0.]])

        expected_sdf_gradient_first_term = np.gradient(live_field)

        for i in range(live_field.shape[0]):
            for j in range(live_field.shape[1]):
                expected_sdf_gradient_second_term = np.array([[1, 0, -i],
                                                              [0, 1, j]])
                expected_sdf_gradient = np.dot(np.array([expected_sdf_gradient_first_term[1][i, j],
                                                         -expected_sdf_gradient_first_term[0][i, j]]),
                                               expected_sdf_gradient_second_term)
                sdf_gradient = sdf_gradient_resp_to_twist(live_field, j, i, twist_vector)

                self.assertTrue(np.allclose(expected_sdf_gradient, sdf_gradient))
