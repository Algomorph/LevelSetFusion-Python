# import unittest
from unittest import TestCase
import numpy as np
import math
from math_utils.transformation import twist_vector_to_matrix2d


class MyTestCase(TestCase):

    def test_twist_vector_to_matrix2d01(self):
        vector = np.array([[0],
                           [0],
                           [0]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
        matrix = twist_vector_to_matrix2d(vector)
        self.assertTrue(np.allclose(expected_matrix, matrix))

    def test_twist_vector_to_matrix2d02(self):
        vector = np.array([[1],
                           [-1],
                           [0]])
        expected_matrix = np.array([[1, 0, 1],
                                    [0, 1, -1],
                                    [0, 0, 1]])
        matrix = twist_vector_to_matrix2d(vector)
        self.assertTrue(np.allclose(expected_matrix, matrix))

    def test_twist_vector_to_matrix2d03(self):
        vector = np.array([[0],
                           [0],
                           [math.pi]])
        expected_matrix = np.array([[-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 1]])
        matrix = twist_vector_to_matrix2d(vector)
        self.assertTrue(np.allclose(expected_matrix, matrix))

    def test_twist_vector_to_matrix2d04(self):
        vector = np.array([[0],
                           [0],
                           [2*math.pi]])
        expected_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
        matrix = twist_vector_to_matrix2d(vector)
        self.assertTrue(np.allclose(expected_matrix, matrix))

    def test_twist_vector_to_matrix2d05(self):
        vector = np.array([[80],
                           [100000],
                           [math.pi/3]])
        expected_matrix = np.array([[1/2, -math.sqrt(3)/2, 80],
                                    [math.sqrt(3)/2, 1/2, 100000],
                                    [0, 0, 1]])
        matrix = twist_vector_to_matrix2d(vector)
        self.assertTrue(np.allclose(expected_matrix, matrix))

    def test_twist_vector_to_matrix2d06(self):
        vector = np.array([[100],
                           [-100],
                           [math.pi/3]])
        expected_matrix = np.array([[1/2, math.sqrt(3)/2, -100],
                                    [-math.sqrt(3)/2, 1/2, 100],
                                    [0, 0, 1]])
        matrix = twist_vector_to_matrix2d(-vector)
        self.assertTrue(np.allclose(expected_matrix, matrix))