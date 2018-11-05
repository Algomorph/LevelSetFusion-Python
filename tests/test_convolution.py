#  ================================================================
#  Created by Gregory Kramida on 11/5/18.
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
import sobolev_smoothing as ss


class ConvolutionTestCase(TestCase):
    def test_convolution(self):
        field = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9], dtype=np.float32).reshape(3, 3)
        vector_field = np.dstack([field] * 2)
        vector_field_out = vector_field.copy()
        kernel = np.array([1, 2, 3])
        ss.convolve_with_sobolev_smoothing_kernel(vector_field_out, np.flip(kernel))
        expected_output = np.dstack([np.array([[85, 168, 99],
                                               [124, 228, 132],
                                               [67, 120, 69]], dtype=np.float32)] * 2)

        self.assertTrue(np.allclose(vector_field_out, expected_output))
