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
import argparse
from enum import Enum

# local
from ext_argparse.argument import Argument
from tsdf import generation as tsdf
import experiment.build_hierarchical_optimizer_helper as build_opt


class Arguments(Enum):
    dataset_number = Argument(arg_type=int, default=1)
    max_warp_update_threshold = Argument(arg_type=float, default=0.01)
    smoothing_coefficient = Argument(arg_type=float, default=0.5)
    max_iteration_count = Argument(arg_type=int, default=1000)
    generation_method = Argument(arg_type=str, default="BASIC")
    implementation_language = Argument(arg_type=str, default="CPP")
    stop_before_index = Argument(arg_type=int, default=10000000)
    start_from_index = Argument(arg_type=int, default=0)
    output_path = Argument(arg_type=str, default="output/ho")

    rate = Argument(arg_type=float, default=0.1)
    data_term_amplifier = Argument(arg_type=float, default=1.0)
    tikhonov_strength = Argument(arg_type=float, default=0.2)
    kernel_size = Argument(arg_type=int, default=7)
    kernel_strength = Argument(arg_type=float, default=0.1, shorthand="-kst")

    # flags
    tikhonov_term_enabled = Argument(action="store_true", default=False, arg_type='bool_flag')
    gradient_kernel_enabled = Argument(action="store_true", default=False, arg_type='bool_flag')

    analyze_only = Argument(action="store_true", default=False, arg_type='bool_flag',
                            arg_help="Skip anything by the final analysis (and only do that if corresponding output"
                                     " file is availalbe). Supersedes any other option that deals with data"
                                     " generation / optimization.")

    bad_cases_only = Argument(action="store_true", default=False, arg_type='bool_flag')
    generate_data = Argument(action="store_true", default=False, arg_type='bool_flag')
    skip_optimization = Argument(action="store_true", default=False, arg_type='bool_flag')
    save_initial_fields_during_generation = Argument(action="store_true", default=False, arg_type='bool_flag')
    save_initial_and_final_fields = Argument(action="store_true", default=False, arg_type='bool_flag',
                                             arg_help="save the initial canoinical & live and final live field during"
                                                      " the optimization")
    save_telemetry = Argument(action="store_true", default=False, arg_type='bool_flag')


def post_process_enum_args(args):
    args.generation_method = tsdf.GenerationMethod.__dict__[args.generation_method]
    args.implementation_language = build_opt.ImplementationLanguage.__dict__[args.implementation_language]
