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
from optimizer2d import zero_warps_for_truncated_values


class DataTermTest(TestCase):
    def test_data_term01(self):
        warped_live_field = np.array([[0.3, 0.4],
                                      [0.8, 0.8]], dtype=np.float32)
        canonical_field = np.array([[0.5, 0.5],
                                    [1.0, 0.8]], dtype=np.float32)

        (live_gradient_y, live_gradient_x) = np.gradient(warped_live_field)

        expected_gradient_out = np.array([[[-.2, -1], [-.1, -.4]],
                                          [[0.0, -1.0], [0.0, 0.0]]], dtype=np.float32)
        data_gradient_out, energy_out = \
            dt.data_term_gradient_direct(warped_live_field, canonical_field, live_gradient_x, live_gradient_y)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, 0.045)

        data_gradient_out, energy_out = \
            dt.data_term_gradient_vectorized(warped_live_field, canonical_field, live_gradient_x, live_gradient_y)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, 0.045)

    def test_data_term02(self):
        warped_live_field = np.array([[0.3, 0.4],
                                      [-0.8, 1.0]], dtype=np.float32)
        canonical_field = np.array([[0.5, -0.3],
                                    [1.0, -1.0]], dtype=np.float32)

        (live_gradient_y, live_gradient_x) = np.gradient(warped_live_field)

        expected_gradient_out = np.array([[[-.2, 2.2], [0.7, 4.2]],
                                          [[-32.4, 19.8], [0.0, 0.0]]], dtype=np.float32)
        data_gradient_out, energy_out = \
            dt.data_term_gradient_direct(warped_live_field, canonical_field, live_gradient_x, live_gradient_y)


        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out, atol=1e-6))
        self.assertAlmostEqual(energy_out, 1.885, places=6)

        data_gradient_out, energy_out = \
            dt.data_term_gradient_vectorized(warped_live_field, canonical_field, live_gradient_x, live_gradient_y)

        zero_warps_for_truncated_values(warped_live_field, canonical_field, data_gradient_out)

        self.assertTrue(np.allclose(data_gradient_out, expected_gradient_out))
        self.assertAlmostEqual(energy_out, 3.885, places=6)
