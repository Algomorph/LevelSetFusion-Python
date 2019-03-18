#  ================================================================
#  Created by Gregory Kramida on 3/15/19.
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

# standard library
from enum import Enum
# requires Python 3.3+

# local
import nonrigid_opt.hierarchical.hierarchical_optimizer2d as ho_py
import nonrigid_opt.hierarchical.hierarchical_optimization_visualizer as hov_py
# has to be built & installed first (git submodule in cpp folder or http://github/Algomorph/LevelSetFusion-CPP)
import level_set_fusion_optimization as ho_cpp
import nonrigid_opt.slavcheva.sobolev_filter as sob


class ImplementationLanguage(Enum):
    PYTHON = 0
    CPP = 1


class HierarchicalOptimizer2dSharedParameters:
    def __init__(self, tikhonov_term_enabled=False,
                 gradient_kernel_enabled=False,

                 maximum_chunk_size=8,
                 rate=0.2,
                 maximum_iteration_count=100,
                 maximum_warp_update_threshold=0.1,

                 data_term_amplifier=1.0,
                 tikhonov_strength=0.0,
                 kernel=sob.generate_1d_sobolev_kernel(size=7, strength=0.1)):
        self.tikhonov_term_enabled = tikhonov_term_enabled
        self.gradient_kernel_enabled = gradient_kernel_enabled

        self.maximum_chunk_size = maximum_chunk_size
        self.rate = rate
        self.maximum_iteration_count = maximum_iteration_count
        self.maximum_warp_update_threshold = maximum_warp_update_threshold

        self.data_term_amplifier = data_term_amplifier
        self.tikhonov_strength = tikhonov_strength
        self.kernel = kernel


def make_common_hierarchical_optimizer2d_visualization_parameters(out_path="out/ho"):
    visualization_parameters = hov_py.HierarchicalOptimizer2dVisualizer.Parameters(
        out_path=out_path,
        save_live_progression=True,
        save_initial_fields=True,
        save_final_fields=True,
        save_warp_field_progression=True,
        save_data_gradients=True,
        save_tikhonov_gradients=False
    )
    return visualization_parameters


def make_common_hierarchical_optimizer2d_py_verbosity_parameters():
    verbosity_parameters = ho_py.HierarchicalOptimizer2d.VerbosityParameters(
        print_max_warp_update=True,
        print_iteration_data_energy=True,
        print_iteration_tikhonov_energy=True,
    )
    return verbosity_parameters


def make_hierarchical_optimizer2d(implementation_language=ImplementationLanguage.CPP,
                                  shared_parameters=HierarchicalOptimizer2dSharedParameters(),
                                  verbosity_parameters_cpp=ho_cpp.HierarchicalOptimizer2d.VerbosityParameters(),
                                  verbosity_parameters_py=
                                  make_common_hierarchical_optimizer2d_py_verbosity_parameters(),
                                  visualization_parameters_py=
                                  make_common_hierarchical_optimizer2d_visualization_parameters(),
                                  logging_parameters_cpp=ho_cpp.HierarchicalOptimizer2d.LoggingParameters(
                                      collect_per_level_convergence_reports=True
                                  )):
    if implementation_language == ImplementationLanguage.CPP:
        return make_cpp_optimizer(shared_parameters, verbosity_parameters_cpp, logging_parameters_cpp)
    elif implementation_language == ImplementationLanguage.PYTHON:
        return make_python_optimizer(shared_parameters, verbosity_parameters_py, visualization_parameters_py)
    else:
        raise ValueError("Unsupported ImplementationLanguage: " + str(implementation_language))


def make_python_optimizer(shared_parameters=HierarchicalOptimizer2dSharedParameters(),
                          verbosity_parameters=make_common_hierarchical_optimizer2d_py_verbosity_parameters(),
                          visualization_parameters=make_common_hierarchical_optimizer2d_visualization_parameters()):
    optimizer = ho_py.HierarchicalOptimizer2d(
        tikhonov_term_enabled=shared_parameters.tikhonov_term_enabled,
        gradient_kernel_enabled=shared_parameters.gradient_kernel_enabled,

        maximum_chunk_size=shared_parameters.maximum_chunk_size,
        rate=shared_parameters.rate,
        maximum_iteration_count=shared_parameters.maximum_iteration_count,
        maximum_warp_update_threshold=shared_parameters.maximum_warp_update_threshold,

        data_term_amplifier=shared_parameters.data_term_amplifier,
        tikhonov_strength=shared_parameters.tikhonov_strength,
        kernel=shared_parameters.kernel,

        verbosity_parameters=verbosity_parameters,
        visualization_parameters=visualization_parameters
    )
    return optimizer


def make_cpp_optimizer(shared_parameters=HierarchicalOptimizer2dSharedParameters(),
                       verbosity_parameters=ho_cpp.HierarchicalOptimizer2d.VerbosityParameters(),
                       logging_parameters=ho_cpp.HierarchicalOptimizer2d.LoggingParameters(
                           collect_per_level_convergence_reports=True
                       )):
    optimizer = ho_cpp.HierarchicalOptimizer2d(
        tikhonov_term_enabled=shared_parameters.tikhonov_term_enabled,
        gradient_kernel_enabled=shared_parameters.gradient_kernel_enabled,

        maximum_chunk_size=shared_parameters.maximum_chunk_size,
        rate=shared_parameters.rate,
        maximum_iteration_count=shared_parameters.maximum_iteration_count,
        maximum_warp_update_threshold=shared_parameters.maximum_warp_update_threshold,

        data_term_amplifier=shared_parameters.data_term_amplifier,
        tikhonov_strength=shared_parameters.tikhonov_strength,
        kernel=shared_parameters.kernel,

        verbosity_parameters=verbosity_parameters,
        logging_parameters=logging_parameters
    )
    return optimizer
