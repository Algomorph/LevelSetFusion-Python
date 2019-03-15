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

# HNS = hierarchical nonrigid SLAM optimizer

# TODO: rename all references, file names, class names to simply use "hierarchical optimizer" instead of some weird
#  acronym

# stdlib
# libraries
import numpy as np
import scipy.ndimage
# local
from utils.pyramid import ScalarFieldPyramid2d
from utils import field_resampling as resampling
import utils.printing as printing
import math_utils.convolution as convolution
from nonrigid_opt.hierarchical_optimization_visualizer import HNSOVisualizer


class HierarchicalOptimizer2d:
    """
    An alternative approach to level sets which still optimizes on voxel-level, in theory being able to
        deal with topology changes in a straightforward and natural way
    """
    class VerbosityParameters:
        """
        Parameters that control verbosity to stdout.
        Assumes being used in an "immutable" manner, i.e. just a structure that holds values
        """

        def __init__(self, print_max_warp_update=False, print_iteration_data_energy=False,
                     print_iteration_tikhonov_energy=False):
            # per-iteration
            self.print_max_warp_update = print_max_warp_update
            self.print_iteration_data_energy = print_iteration_data_energy
            self.print_iteration_tikhonov_energy = print_iteration_tikhonov_energy
            self.per_iteration_flags = [self.print_max_warp_update,
                                        self.print_iteration_data_energy,
                                        self.print_iteration_tikhonov_energy]
            self.print_per_iteration_info = any(self.per_iteration_flags)

            # per-level
            self.print_per_level_info = True  # TODO: def should be any(self.per_level_flags)

    def __init__(self,
                 tikhonov_term_enabled=True,
                 gradient_kernel_enabled=True,

                 maximum_chunk_size=8,
                 rate=0.1,
                 maximum_iteration_count=100,
                 maximum_warp_update_threshold=0.001,

                 data_term_amplifier=1.0,
                 tikhonov_strength=0.2,
                 kernel=None,

                 verbosity_parameters=None,
                 visualization_parameters=None,
                 ):

        """
        Constructor
        :param maximum_chunk_size: lateral size, in pixels, of the chunk of the lowest (finest) level in the hierarchy
        represented by a single pixel in the highest level in the hierarchy
        :param rate: rate of gradient descent (update = gradient*factor)
        :param tikhonov_strength: strength of the tikhonov (i.e. similarity-based) regularizer for the warps
        :param kernel: kernel used to convolve the gradient at each iteration
        :param maximum_warp_update_threshold: lower threshold on the maximum vector length (after which optimization terminates)
        :param maximum_iteration_count: top threshold on the number of iterations (after which optimization terminates)
        :@type verbosity_parameters: HierarchicalNonrigidSLAMOptimizer2d.VerbosityParameters
        :param verbosity_parameters: parameters for stdout verbosity during optimization
        """
        self.maximum_chunk_size = maximum_chunk_size
        self.rate = rate
        self.data_term_amplifier = data_term_amplifier

        if tikhonov_term_enabled:
            self.tikhonov_strength = tikhonov_strength
            self.tikhonov_term_enabled = True if tikhonov_strength != 0.0 else False
        else:
            self.tikhonov_strength = 0.0
            self.tikhonov_term_enabled = False
        if gradient_kernel_enabled:
            self.gradient_kernel = kernel
            self.gradient_kernel_enabled = True if kernel is not None else False
        else:
            self.gradient_kernel = None
            self.gradient_kernel_enabled = False

        self.maximum_warp_update_threshold = maximum_warp_update_threshold
        self.maximum_iteration_count = maximum_iteration_count
        if verbosity_parameters:
            self.verbosity_parameters = verbosity_parameters
        else:
            self.verbosity_parameters = HierarchicalOptimizer2d.VerbosityParameters()

        if visualization_parameters:
            self.visualization_parameters = visualization_parameters
        else:
            self.visualization_parameters = HNSOVisualizer.Parameters()
        self.visualizer = None
        self.hierarchy_level = 0

    def optimize(self, canonical_field, live_field):
        field_size = canonical_field.shape[0]

        live_gradient_y, live_gradient_x = np.gradient(live_field)

        canonical_pyramid = ScalarFieldPyramid2d(canonical_field, self.maximum_chunk_size)
        live_pyramid = ScalarFieldPyramid2d(live_field, self.maximum_chunk_size)
        live_gradient_x_pyramid = ScalarFieldPyramid2d(live_gradient_x, self.maximum_chunk_size)
        live_gradient_y_pyramid = ScalarFieldPyramid2d(live_gradient_y, self.maximum_chunk_size)
        self.hierarchy_level = 0

        level_count = len(canonical_pyramid.levels)
        warp_field = None

        self.visualizer = HNSOVisualizer(parameters=self.visualization_parameters, field_size=field_size,
                                         level_count=level_count)
        self.visualizer.generate_pre_optimization_visualizations(canonical_field, live_field)

        for canonical_pyramid_level, live_pyramid_level, live_gradient_x_level, live_gradient_y_level \
                in zip(canonical_pyramid.levels,
                       live_pyramid.levels,
                       live_gradient_x_pyramid.levels,
                       live_gradient_y_pyramid.levels):

            if self.hierarchy_level == 0:
                warp_field = np.zeros((canonical_pyramid_level.shape[0], canonical_pyramid_level.shape[1], 2),
                                      dtype=np.float32)
            warp_field = \
                self.__optimize_level(canonical_pyramid_level, live_pyramid_level,
                                      live_gradient_x_level, live_gradient_y_level, warp_field)

            if self.hierarchy_level != level_count - 1:
                warp_field = warp_field.repeat(2, axis=0).repeat(2, axis=1)

            if self.verbosity_parameters.print_per_level_info:
                print("%s[LEVEL %d COMPLETED]%s" % (printing.BOLD_RED, self.hierarchy_level, printing.RESET),
                      end="")

                print()

            self.hierarchy_level += 1
        self.visualizer.generate_post_optimization_visualizations(canonical_field, live_field, warp_field)
        del self.visualizer
        return warp_field

    def __termination_conditions_reached(self, maximum_warp_update, iteration_count):
        return maximum_warp_update < self.maximum_warp_update_threshold or \
               iteration_count >= self.maximum_iteration_count

    def __optimize_level(self, canonical_pyramid_level, live_pyramid_level,
                         live_gradient_x_level, live_gradient_y_level, warp_field):

        maximum_warp_update_length = np.finfo(np.float32).max
        iteration_count = 0

        gradient = np.zeros_like(warp_field)
        normalized_tikhonov_energy = 0
        data_gradient = None
        tikhonov_gradient = None

        while not self.__termination_conditions_reached(maximum_warp_update_length, iteration_count):
            # resample the live & gradients using current warps
            resampled_live = resampling.resample_field(live_pyramid_level, warp_field)
            resampled_live_gradient_x = resampling.resample_field_replacement(live_gradient_x_level, warp_field, 0.0)
            resampled_live_gradient_y = resampling.resample_field_replacement(live_gradient_y_level, warp_field, 0.0)

            # see how badly our sampled values correspond to the canonical values at the same locations
            # data_gradient = (warped_live - canonical) * warped_gradient(live)
            diff = (resampled_live - canonical_pyramid_level)
            data_gradient_x = diff * resampled_live_gradient_x
            data_gradient_y = diff * resampled_live_gradient_y
            # this results in the data term gradient
            data_gradient = np.dstack((data_gradient_x, data_gradient_y))

            if self.tikhonov_term_enabled:
                # calculate tikhonov regularizer (laplacian of the previous update)
                laplace_u = scipy.ndimage.laplace(gradient[:, :, 0])
                laplace_v = scipy.ndimage.laplace(gradient[:, :, 1])
                tikhonov_gradient = np.stack((laplace_u, laplace_v), axis=2)

                if self.verbosity_parameters.print_iteration_tikhonov_energy:
                    warp_gradient_u_x, warp_gradient_u_y = np.gradient(gradient[:, :, 0])
                    warp_gradient_v_x, warp_gradient_v_y = np.gradient(gradient[:, :, 1])
                    gradient_aggregate = \
                        warp_gradient_u_x ** 2 + warp_gradient_v_x ** 2 + \
                        warp_gradient_u_y ** 2 + warp_gradient_v_y ** 2
                    normalized_tikhonov_energy = 1000000 * 0.5 * gradient_aggregate.mean()

                gradient = self.data_term_amplifier * data_gradient - self.tikhonov_strength * tikhonov_gradient
            else:
                gradient = self.data_term_amplifier * data_gradient

            if self.gradient_kernel_enabled:
                convolution.convolve_with_kernel(gradient, self.gradient_kernel)

            # apply gradient-based update to existing warps
            warp_field -= self.rate * gradient

            # perform termination condition updates
            update_lengths = np.linalg.norm(gradient, axis=2)
            max_at = np.unravel_index(np.argmax(update_lengths), update_lengths.shape)
            maximum_warp_update_length = update_lengths[max_at]

            # print output to stdout / log
            if self.verbosity_parameters.print_per_iteration_info:
                print("%s[ITERATION %d COMPLETED]%s" % (printing.BOLD_LIGHT_CYAN, iteration_count, printing.RESET),
                      end="")
                if self.verbosity_parameters.print_max_warp_update:
                    print(" max upd. l.: %f" % maximum_warp_update_length, end="")
                if self.verbosity_parameters.print_iteration_data_energy:
                    normalized_data_energy = 1000000 * (diff ** 2).mean()
                    print(" norm. data energy: %f" % normalized_data_energy, end="")
                if self.verbosity_parameters.print_iteration_tikhonov_energy and self.tikhonov_term_enabled:
                    print(" norm. tikhonov energy: %f" % normalized_tikhonov_energy, end="")
                print()
            inverse_tikhonov_gradient = None if tikhonov_gradient is None else -tikhonov_gradient

            # save & show per-iteration visualizations
            self.visualizer.generate_per_iteration_visualizations(self.hierarchy_level, iteration_count,
                                                                  canonical_pyramid_level, resampled_live,
                                                                  warp_field, data_gradient=data_gradient,
                                                                  inverse_tikhonov_gradient=inverse_tikhonov_gradient)
            iteration_count += 1

        return warp_field
