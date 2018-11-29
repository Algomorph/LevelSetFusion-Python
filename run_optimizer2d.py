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
from tsdf_field_generation import DepthInterpolationMethod

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
    parser.add_argument("-sf", "--start_from", type=int,
                        help="Which sample index to start from for the multiple-test mode, 0-based",
                        default=0)
    parser.add_argument("-dtm", "--data_term_method", type=str, default="basic",
                        help="Method to use for the data term, should be in {basic, thresholded_fdm}")
    parser.add_argument("-o", "--output_path", type=str, default="output/out2D",
                        help="output path for multiple_tests mode")
    parser.add_argument("-c", "--calibration", type=str,
                        default=
                        "/media/algomorph/Data/Reconstruction/real_data/"
                        "KillingFusion Snoopy/snoopy_calib.txt",
                        help="Path to the camera calibration file to use unless using a predefined dataset")
    parser.add_argument("-f", "--frames", type=str,
                        default="/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames",
                        help="Path to the depth frames. Frame image files should have names "
                             "that follow depth_{:0>6d}.png pattern, i.e. depth_000000.png")
    parser.add_argument("-cfi", "--canonical_frame_index", type=int, default=-1,
                        help="Use in single_test mode only. Instead of a predefined dataset, use this index for the"
                             " canonical frame in the folder specified by the --frames/-f argument. Live frame is"
                             " assumed to be this index+1 unless otherwise specified. If this value is changed from"
                             " default, -1, then --pixel_row_index must also be specified.")
    parser.add_argument("-pri", "--pixel_row_index", type=int, default=-1,
                        help="Use in single_test mode only. Uses this specific pixel row (0-based-index) for"
                             " optimization. Has to be used in conjunction with the --canonical_frame_index argument.")
    parser.add_argument("-z", "--z_offset", type=int, default=128,
                        help="The Z (depth) offset for sdf volume SDF relative to image"
                             " plane")
    parser.add_argument("-cfp", "--case_file_path", type=str, default=None,
                        help="input cases file path for multiple_tests_mode")
    parser.add_argument("-oc", "--optimizer_choice", type=str, default="CPP",
                        help="optimizer choice (currently, multiple_tests mode only!), "
                             "must be in {CPP, PYTHON_DIRECT, PYTHON_VECTORIZED}")
    parser.add_argument("-di", "--depth_interpolation_method", type=str, default="none",
                        help="Depth image interpolation method to use when generating SDF. "
                             "Can be one of: {none, bilinear}")

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

    depth_interpolation_method = DepthInterpolationMethod.NONE
    if arguments.depth_interpolation_method == "none":
        depth_interpolation_method = DepthInterpolationMethod.NONE
    elif arguments.depth_interpolation_method == "bilinear":
        depth_interpolation_method = DepthInterpolationMethod.BILINEAR

    if mode == Mode.SINGLE_TEST:
        if arguments.pixel_row_index != -1 or arguments.canonical_frame_index != -1:
            if arguments.pixel_row_index < 0 or arguments.canonical_frame_index < 0:
                raise ValueError("When either pixel_row_index or canonical_frame_index is used, *both* of them must be"
                                 " set to a non-negative integer.")
        perform_single_test(depth_interpolation_method=depth_interpolation_method,
                            out_path=arguments.output_path,
                            frame_path=arguments.frames, calibration_path=arguments.calibration,
                            canonical_frame_index=arguments.canonical_frame_index,
                            pixel_row_index=arguments.pixel_row_index, z_offset=arguments.z_offset)

    if mode == Mode.MULTIPLE_TESTS:
        perform_multiple_tests(arguments.start_from, data_term_method,
                               OptimizerChoice.__dict__[arguments.optimizer_choice],
                               depth_interpolation_method=depth_interpolation_method,
                               out_path=arguments.output_path, input_case_file=arguments.case_file_path,
                               calibration_path=arguments.calibration, frame_path=arguments.frames,
                               z_offset=arguments.z_offset)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
