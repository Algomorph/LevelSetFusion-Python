#  ================================================================
#  Created by Gregory Kramida on 11/30/18.
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
# stdlib
from unittest import TestCase

# test targets
from utils.pyramid import *


class FieldPyramidTest(TestCase):
    def test_construct_scalar_pyramid(self):
        tile = np.array([[1, 2, 5, 6, -1, -2, -5, -6],
                         [3, 4, 7, 8, -3, -4, -7, -8],
                         [-1, -2, -5, -6, 1, 2, 5, 6],
                         [-3, -4, -7, -8, 3, 4, 7, 8],
                         [1, 2, 5, 6, 5, 5, 5, 5],
                         [3, 4, 7, 8, 5, 5, 5, 5],
                         [-1, -2, -5, -6, 5, 5, 5, 5],
                         [-3, -4, -7, -8, 5, 5, 5, 5]], dtype=np.float32)
        field = np.tile(tile, (16, 16))  # results in shape 128 x 128

        pyramid = ScalarFieldPyramid2d(field)
        self.assertEqual(len(pyramid.levels), 4)
        self.assertEqual(pyramid.levels[0].shape, (16, 16))
        self.assertEqual(pyramid.levels[1].shape, (32, 32))
        self.assertEqual(pyramid.levels[2].shape, (64, 64))
        self.assertEqual(pyramid.levels[3].shape, (128, 128))
        l2_00 = tile[0:2, 0:2].mean()
        l2_10 = tile[2:4, 0:2].mean()
        l2_01 = tile[0:2, 2:4].mean()
        l2_11 = tile[2:4, 2:4].mean()
        l2_02 = -l2_00
        l2_12 = -l2_10
        l2_03 = -l2_01
        l2_13 = -l2_11
        self.assertEqual(pyramid.levels[2][0, 0], l2_00)
        self.assertEqual(pyramid.levels[2][1, 0], l2_10)
        self.assertEqual(pyramid.levels[2][0, 1], l2_01)
        self.assertEqual(pyramid.levels[2][1, 1], l2_11)
        self.assertEqual(pyramid.levels[2][0, 0 + 2], l2_02)
        self.assertEqual(pyramid.levels[2][1, 0 + 2], l2_12)
        self.assertEqual(pyramid.levels[2][0, 1 + 2], l2_03)
        self.assertEqual(pyramid.levels[2][1, 1 + 2], l2_13)
        self.assertEqual(pyramid.levels[2][0 + 4, 0], l2_00)
        self.assertEqual(pyramid.levels[2][1 + 4, 0], l2_10)
        self.assertEqual(pyramid.levels[2][0 + 4, 1], l2_01)
        self.assertEqual(pyramid.levels[2][1 + 4, 1], l2_11)
        self.assertEqual(pyramid.levels[2][0, 0 + 4], l2_00)
        self.assertEqual(pyramid.levels[2][1, 0 + 4], l2_10)
        self.assertEqual(pyramid.levels[2][0, 1 + 4], l2_01)
        self.assertEqual(pyramid.levels[2][1, 1 + 4], l2_11)
        l1_00 = np.mean([l2_00, l2_10, l2_01, l2_11])
        l1_01 = np.mean([l2_02, l2_12, l2_03, l2_13])
        l1_10 = l1_00
        l1_11 = 5.0
        self.assertEqual(pyramid.levels[1][0, 0], l1_00)
        self.assertEqual(pyramid.levels[1][1, 0], l1_10)
        self.assertEqual(pyramid.levels[1][0, 1], l1_01)
        self.assertEqual(pyramid.levels[1][1, 1], l1_11)
        self.assertEqual(pyramid.levels[0][0, 0], 5.0/4)


