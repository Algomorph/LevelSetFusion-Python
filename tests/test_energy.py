# import unittest
from unittest import TestCase
import numpy as np


class MyTestCase(TestCase):

    def test_energy01(self):
        canonical_field = np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        canonical_weight = np.ones_like(canonical_field)
        live_field = np.array([[1, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        live_weight = np.ones_like(live_field)
        expected_energy = 1
        energy = np.sum((canonical_field * canonical_weight - live_field * live_weight) ** 2)
        self.assertTrue(energy == expected_energy)

    def test_energy02(self):
        canonical_field = np.array([[1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        canonical_weight = np.ones_like(canonical_field)
        live_field = np.array([[1, 0, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
        live_weight = np.ones_like(live_field)
        expected_energy = 0
        energy = np.sum((canonical_field * canonical_weight - live_field * live_weight) ** 2)
        self.assertTrue(energy == expected_energy)

    def test_energy03(self):
        canonical_field = np.array([[1, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        canonical_weight = np.ones_like(canonical_field)
        live_field = np.array([[0, 0, 0],
                               [0, 0, 0],
                               [0, 0, 1]])
        live_weight = np.ones_like(live_field)
        expected_energy = 2
        energy = np.sum((canonical_field * canonical_weight - live_field * live_weight) ** 2)
        self.assertTrue(energy == expected_energy)

    def test_energy04(self):
        canonical_field = np.array([[0, 0, 0],
                                    [0, 0, 0],
                                    [0, 0, 0]])
        canonical_weight = np.ones_like(canonical_field)
        live_field = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]])
        live_weight = np.ones_like(live_field)
        expected_energy = 9
        energy = np.sum((canonical_field * canonical_weight - live_field * live_weight) ** 2)
        self.assertTrue(energy == expected_energy)

