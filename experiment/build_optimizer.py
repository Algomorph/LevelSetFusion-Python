#  ================================================================
#  Created by Gregory Kramida on 11/26/18.
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

# Contains routines for optimizer construction that wrap the C++ and python versions --
# since those have different arguments / logic

# stdlib
from enum import Enum

# local
from nonrigid_opt.data_term import DataTermMethod
from nonrigid_opt.slavcheva_optimizer2d import ComputeMethod, SlavchevaOptimizer2d, AdaptiveLearningRateMethod
from nonrigid_opt.smoothing_term import SmoothingTermMethod
from nonrigid_opt.sobolev_filter import generate_1d_sobolev_kernel
# has to be compiled and installed first (cpp folder)
import level_set_fusion_optimization as cpp_module


class OptimizerChoice(Enum):
    PYTHON_DIRECT = 0
    PYTHON_VECTORIZED = 1
    CPP = 3


def build_optimizer(optimizer_choice, out_path, field_size, view_scaling_factor=8, max_iterations=100,
                    enable_warp_statistics_logging=False, convergence_threshold=0.1,
                    data_term_method=DataTermMethod.BASIC):
    """
    :type optimizer_choice: OptimizerChoice
    :param optimizer_choice: choice of optimizer
    :param max_iterations: maximum iteration count
    :return: an optimizer constructed using the passed arguments
    """
    if optimizer_choice == OptimizerChoice.PYTHON_DIRECT or optimizer_choice == OptimizerChoice.PYTHON_VECTORIZED:
        compute_method = (ComputeMethod.DIRECT
                          if optimizer_choice == OptimizerChoice.PYTHON_DIRECT
                          else ComputeMethod.VECTORIZED)
        optimizer = SlavchevaOptimizer2d(out_path=out_path,
                                         field_size=field_size,

                                         compute_method=compute_method,

                                         level_set_term_enabled=False,
                                         sobolev_smoothing_enabled=True,

                                         data_term_method=data_term_method,
                                         smoothing_term_method=SmoothingTermMethod.TIKHONOV,
                                         adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                                         data_term_weight=1.0,
                                         smoothing_term_weight=0.2,
                                         isomorphic_enforcement_factor=0.1,
                                         level_set_term_weight=0.2,

                                         maximum_warp_length_lower_threshold=convergence_threshold,
                                         max_iterations=max_iterations,
                                         min_iterations=5,

                                         sobolev_kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),

                                         enable_component_fields=True,
                                         view_scaling_factor=view_scaling_factor)
    elif optimizer_choice == OptimizerChoice.CPP:

        shared_parameters = cpp_module.SharedParameters.get_instance()
        shared_parameters.maximum_iteration_count = max_iterations
        shared_parameters.minimum_iteration_count = 5
        shared_parameters.maximum_warp_length_lower_threshold = convergence_threshold
        shared_parameters.maximum_warp_length_upper_threshold = 10000
        shared_parameters.enable_convergence_status_logging = True
        shared_parameters.enable_warp_statistics_logging = enable_warp_statistics_logging

        sobolev_parameters = cpp_module.SobolevParameters.get_instance()
        sobolev_parameters.set_sobolev_kernel(generate_1d_sobolev_kernel(size=7, strength=0.1))
        sobolev_parameters.smoothing_term_weight = 0.2

        optimizer = cpp_module.SobolevOptimizer2d()
    else:
        raise ValueError("Unrecognized optimizer choice: %s" % str(optimizer_choice))
    return optimizer
