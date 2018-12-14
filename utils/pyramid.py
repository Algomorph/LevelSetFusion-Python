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

# Classes for multi-level hierarchical field representations (and routines constructing them)
# system
import math
# libraries
import numpy as np


def is_power_of_two(number):
    return math.log2(number) % 1 == 0.0


class ScalarFieldPyramid2d:
    def __init__(self, field, maximum_chunk_size=8):
        # check that we can break this field down into tiles
        if not is_power_of_two(field.shape[0]) or not is_power_of_two(field.shape[1]):
            raise ValueError("The argument 'field' must be a 2D numpy array where each dimension is a power of two.")

        if not is_power_of_two(maximum_chunk_size):
            raise ValueError("The argument 'maximum_chunk_size' must be an integer power of 2, i.e. 4, 8, 16, etc.")

        power_of_two_largest_chunk = math.log2(maximum_chunk_size)

        # check that we can get a level with the maximum chunk size
        max_level_count = min(int(math.log(field.shape[0], 2)), int(math.log(field.shape[1], 2)))
        if max_level_count <= power_of_two_largest_chunk:
            raise ValueError("maximum chunk size {:d} is too large for a field of size {:s}"
                             .format(maximum_chunk_size, str(field.shape)))

        level_count = int(max_level_count - power_of_two_largest_chunk)
        last_level = field.copy()
        levels = [last_level]
        for i_level in range(1, level_count):
            reshaped1 = last_level.reshape(last_level.shape[0] // 2, 2, last_level.shape[1] // 2, 2)
            axmoved = np.moveaxis(reshaped1, [0, 1, 2, 3], [0, 2, 1, 3])
            reshaped2 = axmoved.reshape(last_level.shape[0] // 2, last_level.shape[1] // 2, 4)
            current_level = reshaped2.mean(axis=2)
            levels.append(current_level)
            last_level = current_level
        levels.reverse()
        self.levels = levels
