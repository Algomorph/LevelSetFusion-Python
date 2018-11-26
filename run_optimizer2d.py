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

# script that runs two different kinds of experiments -- single-frame (for analyzing single cases in detail)
# and multi-frame) for running the same experiment on multiple data and looking at aggregate statistics

# stdlib
import sys
from enum import Enum
import argparse

# local
from data_term import DataTermMethod
from multiframe_experiment import perform_multiple_tests, OptimizerChoice
from singleframe_experiment import perform_single_test

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


class Mode(Enum):
    SINGLE_TEST = 0
    MULTIPLE_TESTS = 1


def main():
    parser = argparse.ArgumentParser("Level Set Fusion 2D motion tracking optimization simulator")
    # TODO: there is a proper way to split up arguments via argparse so that multiple_tests-only arguments
    # cannot be used for single_test mode
    parser.add_argument("-m", "--mode", type=str, help="Mode: singe_test or multiple_tests", default="single_test")
    parser.add_argument("-sf", "--start_from", type=int, help="Which sample to start from for the multiple-test mode",
                        default=0)
    parser.add_argument("-dtm", "--data_term_method", type=str, default="basic",
                        help="Method to use for the data term, should be in {basic, thresholded_fdm}")
    parser.add_argument("-o", "--output_path", type=str, default="out2D/Snoopy MultiTest",
                        help="output path for multiple_tests mode")
    parser.add_argument("-c", "--calibration", type=str,
                        default=
                        "/media/algomorph/Data/Reconstruction/real_data/"
                        "KillingFusion Snoopy/snoopy_calib.txt",
                        help="Path to the camera calibration file to use for the multiple_tests mode"
                        )
    parser.add_argument("-f", "--frames", type=str,
                        default="/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames",
                        help="Path to the frames for the multiple_tests mode. Frame image files should have names "
                             "that follow depth_{:0>6d}.png pattern, i.e. depth_000000.png")
    parser.add_argument("-z", "--z_offset", type=int, default=128,
                        help="(multiple_tests mode only) the Z (depth) offset for sdf volume SDF relative to image"
                             " plane")
    parser.add_argument("-cfp", "--case_file_path", type=str, default=None,
                        help="input cases file path for multiple_tests_mode")
    parser.add_argument("-oc", "--optimizer_choice", type=str, default="CPP",
                        help="optimizer choice (currently, multiple_tests mode only!), "
                             "must be in {CPP, PYTHON_DIRECT, PYTHON_VECTORIZED}")

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
        perform_multiple_tests(arguments.start_from, data_term_method,
                               OptimizerChoice.__dict__[arguments.optimizer_choice],
                               arguments.output_path, arguments.case_file_path,
                               calibration_path=arguments.calibration, frame_path=arguments.frames,
                               z_offset=arguments.z_offset)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
