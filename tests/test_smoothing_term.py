#  ================================================================
#  Created by Gregory Kramida on 10/17/18.
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
import smoothing_term as st
from utils.tsdf_set_routines import set_zeros_for_values_outside_narrow_band_union


class SmoothingTermTest(TestCase):
    def test_smoothing_term01(self):
        warp_field = np.array([[[0., 0.], [1., 1.]],
                               [[1., 1.], [1., 1.]]], dtype=np.float32)

        live_field = np.array([[0.3, 0.5],
                               [-0.8, 1.]])
        canoincal_field = np.array([[0.5, -0.3],
                                    [1., 0.]])

        expected_gradient_out = np.array([[[-2., -2.], [1., 1.]],
                                          [[1., 1.], [0., 0.]]], dtype=np.float32)
        expected_energy_out = 1.0

        gradient_out, energy_out = st.compute_smoothing_term_gradient_direct(warp_field, live_field, canoincal_field)

        self.assertTrue(np.allclose(gradient_out, expected_gradient_out))
        self.assertTrue(energy_out, expected_gradient_out)

        gradient_out = st.compute_smoothing_term_gradient_vectorized(warp_field)
        energy_out = st.compute_smoothing_term_energy(warp_field, live_field, canoincal_field)

        self.assertTrue(np.allclose(gradient_out, expected_gradient_out))
        self.assertTrue(energy_out, expected_gradient_out)
