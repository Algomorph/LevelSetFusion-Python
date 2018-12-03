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
import field_resampling as resampling


class HierarchicalNonrigidSLAMOptimizer2d:
    def __init__(self,
                 maximum_chunk_size=8,
                 rate=0.1,
                 tikhonov_strength=0.2,
                 maximum_warp_update_threshold=0.001,
                 maximum_iteration_count=100
                 ):
        self.maximum_chunk_size = maximum_chunk_size
        self.rate = rate
        self.tikhonov_strength = tikhonov_strength
        self.maximum_warp_update_threshold = maximum_warp_update_threshold
        self.maximum_iteration_count = maximum_iteration_count

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
        return maximum_warp_update < self.maximum_warp_update_threshold or \
               iteration_count >= self.maximum_iteration_count

    def __optimize_level(self, canonical_pyramid_level, live_pyramid_level,
                         live_gradient_x_level, live_gradient_y_level, warp_field):
        maximum_warp_update = np.iinfo(np.float32).max
        iteration_count = 0
        while not self.__termination_conditions_reached(maximum_warp_update, iteration_count):
            # resample the live & gradients using current warps
            resampled_live = resampling.resample_field(live_pyramid_level, warp_field)
            resampled_live_gradient_x = resampling.resample_field_replacement(live_gradient_x_level, warp_field, 0.0)
            resampled_live_gradient_y = resampling.resample_field_replacement(live_gradient_y_level, warp_field, 0.0)

            # see how badly our sampled values correspond to the canonical values at the same locations
            # (warped_live - canonical) * warped_gradient(live)
            diff = (resampled_live - canonical_pyramid_level)
            data_gradient_x = diff * resampled_live_gradient_x
            data_gradient_y = diff * resampled_live_gradient_y

            # TODO: Tikhonov & Sobolev regularization

            # apply gradient-based update to existing warps
            gradient_x = data_gradient_x
            gradient_y = data_gradient_y
            warp_update = self.rate * np.dstack((gradient_x, gradient_y))
            warp_field -= warp_update

            # perform termination condition updates
            update_lengths = np.linalg.norm(warp_update, axis=2)
            max_at = np.unravel_index(np.argmax(update_lengths), update_lengths.shape)
            maximum_warp_update = update_lengths[max_at]
            iteration_count += 1

        return warp_field
