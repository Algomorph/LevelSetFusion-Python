#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 11/14/17.
#  Copyright (c) 2017 Gregory Kramida
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
import sys
from enum import Enum
import argparse

# libraries
import numpy as np

# local
from data_term import DataTermMethod
from dataset import datasets, DataToUse
from smoothing_term import SmoothingTermMethod
from snoopy_multiframe_experiment import perform_multiple_tests
from tsdf_field_generation import generate_initial_orthographic_2d_tsdf_fields
from optimizer2d import Optimizer2D, AdaptiveLearningRateMethod
from sobolev_filter import generate_1d_sobolev_kernel
from vizualization import visualize_and_save_initial_fields, visualize_final_fields

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def perform_single_test():
    visualize_and_save_initial_and_final_fields = True
    field_size = 128
    default_value = 1
    out_path = "out2D"
    data_to_use = DataToUse.REAL3D_SNOOPY_SET02

    if data_to_use == DataToUse.GENEREATED2D:
        live_field, canonical_field = \
            generate_initial_orthographic_2d_tsdf_fields(field_size=field_size,
                                                         live_smoothing_kernel_size=0,
                                                         canonical_smoothing_kernel_size=0,
                                                         default_value=default_value)
    else:
        live_field, canonical_field = datasets[data_to_use].generate_2d_sdf_fields(default_value)
        field_size = datasets[data_to_use].field_size

    warp_field = np.zeros((field_size, field_size, 2), dtype=np.float32)
    view_scaling_factor = 1024 // field_size

    if visualize_and_save_initial_and_final_fields:
        visualize_and_save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor)

    optimizer = Optimizer2D(out_path=out_path,
                            field_size=field_size,
                            default_value=default_value,

                            level_set_term_enabled=False,
                            sobolev_smoothing_enabled=True,

                            data_term_method=DataTermMethod.BASIC,
                            smoothing_term_method=SmoothingTermMethod.TIKHONOV,
                            adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                            data_term_weight=1.0,
                            smoothing_term_weight=0.2,
                            isomorphic_enforcement_factor=0.1,
                            level_set_term_weight=0.2,

                            maximum_warp_length_lower_threshold=0.1,
                            max_iterations=100,

                            sobolev_kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),

                            enable_component_fields=True,
                            view_scaling_factor=view_scaling_factor)

    optimizer.optimize(live_field, canonical_field, warp_field)
    optimizer.plot_logged_sdf_and_warp_magnitudes()
    optimizer.plot_logged_energies_and_max_warps()

    if visualize_and_save_initial_and_final_fields:
        visualize_final_fields(canonical_field, live_field, view_scaling_factor)


class Mode(Enum):
    SINGLE_TEST = 0
    MULTIPLE_TESTS = 1


def main():
    parser = argparse.ArgumentParser("Level Set Fusion 2D motion tracking optimization simulator")
    parser.add_argument("-m", "--mode", type=str, help="Mode: singe_test or multiple_tests", default="single_test")
    parser.add_argument("-sf", "--start_from", type=int, help="Which sample to start from for the multiple-test mode",
                        default=0)
    parser.add_argument("-dtm", "--data_term_method", type=str, default="basic",
                        help="Method to use for the data term, should be in {basic, thresholded_fdm}")
    parser.add_argument("-o", "--output_path", type=str, default="out2D/Snoopy MultiTest",
                        help="output path for multiple tests mode")

    arguments = parser.parse_args()
    mode = Mode.SINGLE_TEST

    if "mode" in arguments:
        mode_argument = arguments.mode
        if mode_argument == "single_test":
            mode = Mode.SINGLE_TEST
        elif mode_argument == "multiple_tests":
            mode = Mode.MULTIPLE_TESTS
        else:
            print("Invalid program command argument:" +
                  " mode should be \"single_test\" or \"multiple_tests\", got \"{:s}\"".format(mode_argument))
    data_term_method = DataTermMethod.BASIC
    if arguments.data_term_method == "basic":
        data_term_method = DataTermMethod.BASIC
    elif arguments.data_term_method == "thresholded_fdm":
        data_term_method = DataTermMethod.THRESHOLDED_FDM
    else:
        print("Invalid program command argument:" +
              " data_term_method (dtm) should be \"basic\" or \"thresholded_fdm\", got \"{:s}\""
              .format(arguments.data_term_method))

    if mode == Mode.SINGLE_TEST:
        perform_single_test()
    if mode == Mode.MULTIPLE_TESTS:
        perform_multiple_tests(arguments.start_from, data_term_method, arguments.output_path)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
