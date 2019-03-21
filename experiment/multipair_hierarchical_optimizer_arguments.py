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

    # flags
    analyze_only = Argument(action="store_true", default=False, arg_type='bool_flag',
                            arg_help="Skip anything by the final analysis (and only do that if corresponding output"
                                     " file is availalbe). Supersedes any other option that deals with data"
                                     " generation / optimization.")
    generate_data = Argument(action="store_true", default=False, arg_type='bool_flag')
    skip_optimization = Argument(action="store_true", default=False, arg_type='bool_flag')
    save_initial_fields_during_generation = Argument(action="store_true", default=False, arg_type='bool_flag')
    save_final_fields = Argument(action="store_true", default=False, arg_type='bool_flag')
    save_telemetry = Argument(action="store_true", default=False, arg_type='bool_flag')


def post_process_enum_args(args):
    args.generation_method = tsdf.GenerationMethod.__dict__[args.generation_method]
    args.implementation_language = build_opt.ImplementationLanguage.__dict__[args.implementation_language]


def legacy_process_args():
    parser = argparse.ArgumentParser(
        "Runs 2D hierarchical optimizer on TSDF inputs generated from frame-pairs "
        "& random pixel rows from these. Alternatively, generates the said data or "
        "loads it from a folder from further re-use.")
    # TODO figure out how to have positional and optional arguments share a destination
    parser.add_argument("--dataset_number", "-dn", type=int, default=1)
    parser.add_argument("--max_warp_update_threshold", "-mwut", type=float, default=0.01)
    parser.add_argument("--smoothing_coefficient", "-sc", type=float, default=0.5)
    parser.add_argument("--max_iteration_count", "-mic", type=int, default=1000)
    parser.add_argument("--generation_method", "-gm", type=str, default="BASIC")
    parser.add_argument("--implementation_language", "-im", type=str,
                        default="CPP")
    parser.add_argument("--stop_before_index", "-sbi", type=int, default=10000000)
    parser.add_argument("--start_from_index", "-sfi", type=int, default=0)
    parser.add_argument("--output_path", "-o", type=str, default="output/ho")

    # flags
    parser.add_argument("--analyze_only", "-ao", action="store_true", default=False,
                        help="Skip anything by the final analysis (and only do that if corresponding output file"
                             " is availalbe). Supersedes any other option that deals with data generation /"
                             " optimization.")
    parser.add_argument("--generate_data", "-gd", action="store_true", default=False)
    parser.add_argument("--skip_optimization", "-so", action="store_true", default=False)
    parser.add_argument("--save_initial_fields_during_generation", "-sifdg", action="store_true", default=False)
    parser.add_argument("--save_final_fields", "-sff", action="store_true", default=False)
    parser.add_argument("--save_telemetry", "-st", action="store_true", default=False)

    args = parser.parse_args()
    post_process_enum_args(args)
    return args
