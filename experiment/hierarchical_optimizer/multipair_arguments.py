#  ================================================================
#  Created by Gregory Kramida on 3/21/19.
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
from enum import Enum

# local
from ext_argparse.argument import Argument
from tsdf import generation as tsdf
import experiment.hierarchical_optimizer.build_helper as build_opt

# NB: needs to be compiled and installed / added to PYTHONPATH first
import level_set_fusion_optimization as cpp_module


class Arguments(Enum):
    # optimizer settings
    tikhonov_term_enabled = Argument(action="store_true", default=False, arg_type='bool_flag')
    gradient_kernel_enabled = Argument(action="store_true", default=False, arg_type='bool_flag')

    max_warp_update_threshold = Argument(arg_type=float, default=0.01)
    max_iteration_count = Argument(arg_type=int, default=1000)

    rate = Argument(arg_type=float, default=0.1)
    data_term_amplifier = Argument(arg_type=float, default=1.0)
    tikhonov_strength = Argument(arg_type=float, default=0.2)
    kernel_size = Argument(arg_type=int, default=7)
    kernel_strength = Argument(arg_type=float, default=0.1, shorthand="-kst")
    resampling_strategy = Argument(arg_type=str, default="NEAREST_AND_AVERAGE",
                                   arg_help="Strategy for upsampling the warps and downsampling the pyramid"
                                            "in the C++ version of the optimizer, can be "
                                            "either NEAREST_AND_AVERAGE or LINEAR")

    # data generation settings
    generation_method = Argument(arg_type=str, default="BASIC")
    smoothing_coefficient = Argument(arg_type=float, default=0.5)

    # other experiment settings
    dataset_number = Argument(arg_type=int, default=1)
    implementation_language = Argument(arg_type=str, default="CPP")
    stop_before_index = Argument(arg_type=int, default=10000000)
    start_from_index = Argument(arg_type=int, default=0)
    output_path = Argument(arg_type=str, default="output/ho")
    generation_case_file = \
        Argument(arg_type=str, default=None,
                 arg_help="Generate data for the set of frames & pixel rows specified in this .csv file."
                          " Format is <frame_index>,<pixel row index>,<focus coordinate x>, "
                          "<focus coordinate y>.")
    optimization_case_file = \
        Argument(arg_type=str, default=None,
                 arg_help="Run optimizer only on the set of frames & pixel rows specified in this .csv file "
                          "(assuming they are also present in the specified dataset)."
                          " Format is <frame_index>,<pixel row index>,<focus coordinate x>, "
                          "<focus coordinate y>.")
    series_result_subfolder = Argument(arg_type=str, default=None,
                                       arg_help="Additional subfolder name to append to the output directory (useful "
                                                "when saving results for a whole series)")

    # other experiment flags
    analyze_only = Argument(action="store_true", default=False, arg_type='bool_flag',
                            arg_help="Skip anything by the final analysis (and only do that if corresponding output"
                                     " file is available). Supersedes any other option that deals with data"
                                     " generation / optimization.")
    generate_data = Argument(action="store_true", default=False, arg_type='bool_flag')
    skip_optimization = Argument(action="store_true", default=False, arg_type='bool_flag')
    save_initial_fields_during_generation = Argument(action="store_true", default=False, arg_type='bool_flag')
    save_initial_and_final_fields = Argument(action="store_true", default=False, arg_type='bool_flag',
                                             arg_help="save the initial canonical & live and final live field during"
                                                      " the optimization")
    save_telemetry = Argument(action="store_true", default=False, arg_type='bool_flag')
    convert_telemetry = Argument(action="store_true", default=False, arg_type='bool_flag',
                                 arg_help="Convert telemetry to videos")


def post_process_enum_args(args):
    Arguments.generation_method.v = \
        args.generation_method = tsdf.GenerationMethod.__dict__[args.generation_method]
    Arguments.implementation_language.v = args.implementation_language = \
        build_opt.ImplementationLanguage.__dict__[args.implementation_language]
    resampling_strategies = cpp_module.HierarchicalOptimizer2d.ResamplingStrategy.__dict__['values']
    Arguments.resampling_strategy.v = args.resampling_strategy = \
        [val for key, val in resampling_strategies.items() if val.name == Arguments.resampling_strategy.v][0]
