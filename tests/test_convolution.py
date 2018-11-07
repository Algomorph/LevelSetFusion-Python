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
import math_utils.convolution as mc


class ConvolutionTestCase(TestCase):
    def test_convolution(self):
        field = np.array([1, 4, 7, 2, 5, 8, 3, 6, 9], dtype=np.float32).reshape(3, 3)
        vector_field = np.dstack([field] * 2)
        kernel = np.array([1, 2, 3])
        mc.convolve_with_kernel_preserve_zeros(vector_field, np.flip(kernel))
        expected_output = np.dstack([np.array([[85, 168, 99],
                                               [124, 228, 132],
                                               [67, 120, 69]], dtype=np.float32)] * 2)

        self.assertTrue(np.allclose(vector_field, expected_output))

    def test_convolution2(self):
        # corresponds to convolution_test02 in C++

        vector_field = np.array([[[0., 0.],
                                  [0., 0.],
                                  [-0.35937524, -0.18750024],
                                  [-0.13125, -0.17500037]],

                                 [[0., 0.],
                                  [-0.4062504, -0.4062496],
                                  [-0.09375, -0.1874992],
                                  [-0.04375001, -0.17499907]],

                                 [[0., 0.],
                                  [-0.65624946, -0.21874908],
                                  [-0.09375, -0.1499992],
                                  [-0.04375001, -0.21874908]],

                                 [[0., 0.],
                                  [-0.5312497, -0.18750025],
                                  [-0.09374999, -0.15000032],
                                  [-0.13125001, -0.2625004]]], dtype=np.float32)
        kernel = np.array([0.06742075, 0.99544406, 0.06742075], dtype=np.float32)
        expected_output = np.array([[[0., 0.],
                                     [0., 0.],
                                     [-0.37325418, -0.2127664],
                                     [-0.15753812, -0.19859032]],

                                    [[0., 0.],
                                     [-0.45495194, -0.43135524],
                                     [-0.15728821, -0.25023925],
                                     [-0.06344876, -0.21395193]],

                                    [[0., 0.],
                                     [-0.7203466, -0.26821017],
                                     [-0.15751792, -0.20533602],
                                     [-0.06224135, -0.25772366]],

                                    [[0., 0.],
                                     [-0.57718134, -0.21122558],
                                     [-0.14683422, -0.19089347],
                                     [-0.13971107, -0.2855439]]], dtype=np.float32)
        mc.convolve_with_kernel_preserve_zeros(vector_field, np.flip(kernel))
        self.assertTrue(np.allclose(vector_field, expected_output))

    def test_convolution3(self):
        # corresponds to convolution_test02 in C++

        vector_field = np.array([[[0., 0.],
                                  [0., 0.],
                                  [-0.35937524, -0.18750024],
                                  [-0.13125, -0.17500037]],

                                 [[0., 0.],
                                  [-0.4062504, -0.4062496],
                                  [-0.09375, -0.1874992],
                                  [-0.04375001, -0.17499907]],

                                 [[0., 0.],
                                  [-0.65624946, -0.21874908],
                                  [-0.09375, -0.1499992],
                                  [-0.04375001, -0.21874908]],

                                 [[0., 0.],
                                  [-0.5312497, -0.18750025],
                                  [-0.09374999, -0.15000032],
                                  [-0.13125001, -0.2625004]]], dtype=np.float32)
        kernel = np.array([0.06742075, 0.99544406, 0.06742075], dtype=np.float32)
        expected_output = np.array([[[0., 0.],
                                     [0., 0.],
                                     [-0.35937524, -0.18750024],
                                     [-0.13125, -0.17500037]],

                                    [[0., 0.],
                                     [-0.4062504, -0.4062496],
                                     [-0.09375, -0.1874992],
                                     [-0.04375001, -0.17499907]],

                                    [[0., 0.],
                                     [-0.65624946, -0.21874908],
                                     [-0.09375, -0.1499992],
                                     [-0.04375001, -0.21874908]],

                                    [[0., 0.],
                                     [-0.5312497, -0.18750025],
                                     [-0.09374999, -0.15000032],
                                     [-0.13125001, -0.2625004]]], dtype=np.float32)
        mc.convolve_with_kernel_y(vector_field, kernel)
        self.assertTrue(np.allclose(vector_field, expected_output))
