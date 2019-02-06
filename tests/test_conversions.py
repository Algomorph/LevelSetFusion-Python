#  ================================================================
#  Created by Gregory Kramida on 2/6/19.
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
# local
# C++ extension
import level_set_fusion_optimization as cpp_extension


class CoonversionTest(TestCase):
    def test_tensor_f3_basic(self):
        t = np.array([[[1, 2, 3, 4],
                       [5, 6, 7, 8]],
                      [[9, 10, 11, 12],
                       [13, 14, 15, 16]]], dtype=np.float32)

        t2 = cpp_extension.return_input_f3(t)
        self.assertTrue(np.allclose(t, t2))

        t3 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
                       [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]],
                       [[25, 26, 27], [28, 29, 30], [31, 32, 33], [34, 35, 36]]]
                      , dtype=np.float32)

        t4 = cpp_extension.return_input_f3(t3)
        self.assertTrue(np.allclose(t3, t4))

    def test_tensor_f4_basic(self):
        t = np.arange(1, 49, dtype=np.float32).reshape((2, 4, 2, 3))
        t2 = cpp_extension.return_input_f4(t)
        self.assertTrue(np.allclose(t, t2))

    def test_tensor_f3rm_basic(self):
        t = np.arange(1, 25, dtype=np.float32).reshape((2, 4, 3))
        t2 = cpp_extension.return_tensor_f3rm()
        self.assertTrue(np.allclose(t, t2))

    def test_tensor_f4rm_basic(self):
        t = np.arange(1, 49, dtype=np.float32).reshape((2, 4, 2, 3))
        t2 = cpp_extension.return_tensor_f4rm()
        self.assertTrue(np.allclose(t, t2))

    def test_tensor_f3_scale(self):
        t = np.arange(1, 25, dtype=np.float32).reshape((2, 4, 3))
        factor = 2.5
        t2 = cpp_extension.scale(t, factor)
        self.assertTrue(np.allclose(t * factor, t2))

    def test_tensor_f3_add_constant(self):
        t = np.arange(1, 25, dtype=np.float32).reshape((2, 4, 3))
        constant = 95.2
        t2 = cpp_extension.add_constant(t, constant)
        self.assertTrue(np.allclose(t + constant, t2))

    def test_tensor_f3_add_2_tensors(self):
        t1 = np.arange(1, 25, dtype=np.float32).reshape((2, 4, 3))
        t2 = np.random.rand(2, 4, 3).astype(np.float32) * 15.0
        t3 = cpp_extension.add_tensors(t1, t2)
        self.assertTrue(np.allclose(t1 + t2, t3))
