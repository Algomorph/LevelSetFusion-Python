#  ================================================================
#  Created by Gregory Kramida on 1/11/19.
#  Copyright (c) 2019 Gregory Kramida
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

# stdlib
from unittest import TestCase

# libraries
import numpy as np

# C++ extension
import level_set_fusion_optimization as cpp_extension


class MathTest(TestCase):
    def test_mean_vector_length(self):
        vector_field = np.array([[[0.8562016, 0.876527],
                                  [0.8056713, 0.31369442],
                                  [0.28571403, 0.38419583],
                                  [0.86377007, 0.9078812]],

                                 [[0.12255816, 0.22223428],
                                  [0.4487159, 0.7280231],
                                  [0.61369246, 0.43351218],
                                  [0.3545089, 0.33867624]],

                                 [[0.5658683, 0.53506494],
                                  [0.69546276, 0.9331944],
                                  [0.05706289, 0.06915309],
                                  [0.5286004, 0.9154799]],

                                 [[0.98797816, 0.60008055],
                                  [0.07343615, 0.10326899],
                                  [0.28764063, 0.05625961],
                                  [0.32258928, 0.84611595]]], dtype=np.float32)
        lengths = np.linalg.norm(vector_field, axis=2)
        mean_length = cpp_extension.mean_vector_length(vector_field)
        self.assertAlmostEqual(lengths.mean(), mean_length)
