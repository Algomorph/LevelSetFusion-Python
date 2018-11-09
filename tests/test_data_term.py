#  ================================================================
#  Created by Gregory Kramida on 10/16/18.
#  Copyright (c) 2018 Gregory Kramida
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
import data_term as dt
from utils.tsdf_set_routines import set_zeros_for_values_outside_narrow_band_union


class DataTermTest(TestCase):
    def test_data_term01(self):
        warped_live_field = np.array([[0.3, 0.4],
                                      [0.8, 0.8]], dtype=np.float32)
        canonical_field = np.array([[0.5, 0.5],
                                    [1.0, 0.8]], dtype=np.float32)

        live_gradient_y, live_gradient_x = np.gradient(warped_live_field)

        expected_gradient_out = np.array([[[-.2, -1], [-.1, -.4]],
                                          [[0.0, -1.0], [0.0, 0.0]]], dtype=np.float32)
        expected_energy_out = 0.045
        data_gradient_out, energy_out = \
            dt.compute_data_term_gradient_direct(warped_live_field, canonical_field, live_gradient_x, live_gradient_y)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, expected_energy_out)

        data_gradient_out = dt.compute_data_term_gradient_vectorized(warped_live_field, canonical_field,
                                                                     live_gradient_x, live_gradient_y)

        energy_out = dt.compute_data_term_energy_contribution(warped_live_field, canonical_field)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out, atol=1e-6))
        self.assertAlmostEqual(energy_out, expected_energy_out)

    def test_data_term02(self):
        warped_live_field = np.array([[0.3, 0.4],
                                      [-0.8, 1.0]], dtype=np.float32)
        canonical_field = np.array([[0.5, -0.3],
                                    [1.0, -1.0]], dtype=np.float32)

        live_gradient_y, live_gradient_x = np.gradient(warped_live_field)

        expected_gradient_out = np.array([[[-.2, 2.2], [0.7, 4.2]],
                                          [[-32.4, 19.8], [0.0, 0.0]]], dtype=np.float32)
        expected_energy_out = 1.885
        data_gradient_out, energy_out = \
            dt.compute_data_term_gradient_direct(warped_live_field, canonical_field, live_gradient_x, live_gradient_y)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, expected_energy_out, places=6)

        data_gradient_out = \
            dt.compute_data_term_gradient_vectorized(warped_live_field, canonical_field, live_gradient_x,
                                                     live_gradient_y)
        set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, data_gradient_out)
        energy_out = dt.compute_data_term_energy_contribution(warped_live_field, canonical_field)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, expected_energy_out)

    def test_data_term03(self):
        warped_live_field = np.array([[1., 1., 0.49999955, 0.42499956],
                                      [1., 0.44999936, 0.34999937, 0.32499936],
                                      [1., 0.35000065, 0.25000066, 0.22500065],
                                      [1., 0.20000044, 0.15000044, 0.07500044]], dtype=np.float32)
        canonical_field = np.array([[1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                                    [1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                                    [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                                    [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32)

        live_gradient_y, live_gradient_x = np.gradient(warped_live_field)

        expected_gradient_out = np.array([[[0., 0.],
                                           [0., 0.],
                                           [-0.35937524, -0.18750024],
                                           [-0.13125, -0.17500037], ],

                                          [[0., 0.],
                                           [-0.4062504, -0.4062496],
                                           [-0.09375, -0.1874992],
                                           [-0.04375001, -0.17499907], ],

                                          [[0., 0.],
                                           [-0.65624946, -0.21874908],
                                           [-0.09375, -0.1499992],
                                           [-0.04375001, -0.21874908], ],

                                          [[0., 0.],
                                           [-0.5312497, -0.18750025],
                                           [-0.09374999, -0.15000032],
                                           [-0.13125001, -0.2625004], ]], dtype=np.float32)
        expected_energy_out = 0.13375

        data_gradient_out, energy_out = \
            dt.compute_data_term_gradient_direct(warped_live_field, canonical_field, live_gradient_x, live_gradient_y)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, expected_energy_out)

        data_gradient_out = \
            dt.compute_data_term_gradient_vectorized(warped_live_field, canonical_field, live_gradient_x,
                                                     live_gradient_y)
        set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, data_gradient_out)
        energy_out = dt.compute_data_term_energy_contribution(warped_live_field, canonical_field)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, expected_energy_out)

    def test_data_term04(self):
        # corresponds to test_data_term_gradient01 in C++
        warped_live_field = np.array([[1., 1., 0.49404836, 0.4321034],
                                      [1., 0.44113636, 0.34710377, 0.32715625],
                                      [1., 0.3388706, 0.24753733, 0.22598255],
                                      [1., 0.21407352, 0.16514614, 0.11396749]], dtype=np.float32)
        canonical_field = np.array([[-1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                                    [-1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                                    [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                                    [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32)

        expected_gradient_out = np.array([[[0., 0.],
                                           [-0., -0.],
                                           [-0.33803707, -0.17493579],
                                           [-0.11280416, -0.1911128]],

                                          [[-11.177273, 0.],
                                           [-0.37912706, -0.38390794],
                                           [-0.08383488, -0.1813143],
                                           [-0.03533841, -0.18257865]],

                                          [[-0., 0.],
                                           [-0.6165301, -0.18604389],
                                           [-0.08327565, -0.13422713],
                                           [-0.03793251, -0.18758681]],

                                          [[-0., 0.],
                                           [-0.5805285, -0.17355914],
                                           [-0.0826604, -0.13606551],
                                           [-0.10950545, -0.23967531]]], dtype=np.float32)
        expected_gradient_out_band_union_only = np.array([[[0., 0.],
                                                           [-0., -0.],
                                                           [-0.33803707, -0.17493579],
                                                           [-0.11280416, -0.1911128]],

                                                          [[-0., 0.],
                                                           [-0.37912706, -0.38390794],
                                                           [-0.08383488, -0.1813143],
                                                           [-0.03533841, -0.18257865]],

                                                          [[-0., 0.],
                                                           [-0.6165301, -0.18604389],
                                                           [-0.08327565, -0.13422713],
                                                           [-0.03793251, -0.18758681]],

                                                          [[-0., 0.],
                                                           [-0.5805285, -0.17355914],
                                                           [-0.0826604, -0.13606551],
                                                           [-0.10950545, -0.23967531]]], dtype=np.float32)

        expected_energy_out = 4.142916451210006
        expected_energy_out_band_union_only = 0.14291645121000718

        live_gradient_y, live_gradient_x = np.gradient(warped_live_field)

        data_gradient_out, energy_out = \
            dt.compute_data_term_gradient_direct(warped_live_field, canonical_field, live_gradient_x, live_gradient_y,
                                                 band_union_only=False)
        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, expected_energy_out)
        data_gradient_out, energy_out = \
            dt.compute_data_term_gradient_direct(warped_live_field, canonical_field, live_gradient_x, live_gradient_y,
                                                 band_union_only=True)
        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out_band_union_only))
        self.assertAlmostEqual(energy_out, expected_energy_out_band_union_only)

        data_gradient_out = \
            dt.compute_data_term_gradient_vectorized(warped_live_field, canonical_field, live_gradient_x,
                                                     live_gradient_y)
        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, data_gradient_out)
        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out_band_union_only))
        energy_out = dt.compute_data_term_energy_contribution(warped_live_field, canonical_field, band_union_only=False)
        self.assertAlmostEqual(energy_out, expected_energy_out,places=6)
        energy_out = dt.compute_data_term_energy_contribution(warped_live_field, canonical_field)
        self.assertAlmostEqual(energy_out, expected_energy_out_band_union_only)
