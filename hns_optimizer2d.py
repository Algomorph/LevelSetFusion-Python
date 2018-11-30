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

# HNS = hierarchical nonrigid optimizer
# stdlib
import math
# libraries
import numpy as np
# local
from utils.pyramid import ScalarFieldPyramid2d


class HierarchicalNonrigidSLAMOptimizer2d:
    def __init__(self,
                 maximum_chunk_size=8,
                 rate=0.1,
                 tikhonov_strength=0.2,
                 ):
        self.maximum_chunk_size = maximum_chunk_size
        self.rate = rate
        self.tikhonov_strength = tikhonov_strength

    def optimize(self, canonical_field, live_field):
        live_gradient_y, live_gradient_x = np.gradient(live_field)

        canonical_pyramid = ScalarFieldPyramid2d(canonical_field, self.maximum_chunk_size)
        live_pyramid = ScalarFieldPyramid2d(live_field, self.maximum_chunk_size)
        live_gradient_x_pyramid = ScalarFieldPyramid2d(live_gradient_x, self.maximum_chunk_size)
        live_gradient_y_pyramid = ScalarFieldPyramid2d(live_gradient_y, self.maximum_chunk_size)
        i_level = 0

        level_count = len(canonical_pyramid.levels)
        warp_field = None

        for canonical_pyramid_level, live_pyramid_level, live_gradient_x_level, live_gradient_y_level \
                in zip(canonical_pyramid.levels,
                       live_pyramid.levels,
                       live_gradient_x_pyramid.levels,
                       live_gradient_y_pyramid.levels):
            if i_level == 0:
                warp_field = np.zeros((canonical_pyramid_level.shape[0], canonical_pyramid_level.shape[1], 2),
                                      dtype=np.float32)
            warp_field = \
                self.__optimize_level(canonical_pyramid_level, live_pyramid_level,
                                      live_gradient_x_level, live_gradient_y_level, warp_field)

            if i_level != level_count - 1:
                warp_field = warp_field.repeat(2, axis=0).repeat(2, axis=1)
            i_level += 1

    def __termination_conditions_reached(self, maximum_warp_update, iteration_count):
        # TODO: check for termination criteria
        raise NotImplementedError()
        return False

    def __optimize_level(self, canonical_pyramid_level, live_pyramid_level,
                         live_gradient_x_level, live_gradient_y_level, warp_field):
        maximum_warp_update = np.iinfo(np.float32).max
        iteration_count = 0
        while not self.__termination_conditions_reached(maximum_warp_update, iteration_count):
            # ==== data term ===
            pass

            # TODO: perform iteration

        raise NotImplementedError()
        return warp_field
