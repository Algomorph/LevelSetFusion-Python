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
import os
import os.path
import sys
from enum import Enum
import gc
import argparse

# libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt

# local
from data_term import DataTermMethod
from dataset import datasets, DataToUse, Dataset
from smoothing_term import SmoothingTermMethod
from tsdf_field_generation import generate_initial_orthographic_2d_tsdf_fields
from optimizer2d import Optimizer2D, AdaptiveLearningRateMethod
from sobolev_filter import generate_1d_sobolev_kernel
from vizualization import process_cv_esc, sdf_field_to_image, mark_focus_coordinate_on_sdf_image
from printing import *

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def visualize_and_save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor=8):
    canonical_visualized = sdf_field_to_image(canonical_field, scale=view_scaling_factor)
    canonical_visualized = mark_focus_coordinate_on_sdf_image(canonical_visualized)
    canonical_visualized_unscaled = sdf_field_to_image(canonical_field, scale=1)
    cv2.imshow("canonical SDF", canonical_visualized)
    cv2.imwrite(os.path.join(out_path, 'unscaled_initial_canonical.png'), canonical_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, 'initial_canonical.png'), canonical_visualized)
    process_cv_esc()
    live_visualized = sdf_field_to_image(live_field, scale=view_scaling_factor)
    live_visualized = mark_focus_coordinate_on_sdf_image(live_visualized)
    live_visualized_unscaled = sdf_field_to_image(live_field, scale=1)
    cv2.imwrite(os.path.join(out_path, "unscaled_initial_live.png"), live_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, "initial_live.png"), live_visualized)
    cv2.imshow("live SDF", live_visualized)
    process_cv_esc()
    cv2.destroyAllWindows()


def save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor=8):
    canonical_visualized = sdf_field_to_image(canonical_field, scale=view_scaling_factor)
    canonical_visualized = mark_focus_coordinate_on_sdf_image(canonical_visualized)
    canonical_visualized_unscaled = sdf_field_to_image(canonical_field, scale=1)

    cv2.imwrite(os.path.join(out_path, 'unscaled_initial_canonical.png'), canonical_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, 'initial_canonical.png'), canonical_visualized)

    live_visualized = sdf_field_to_image(live_field, scale=view_scaling_factor)
    live_visualized = mark_focus_coordinate_on_sdf_image(live_visualized)
    live_visualized_unscaled = sdf_field_to_image(live_field, scale=1)
    cv2.imwrite(os.path.join(out_path, "unscaled_initial_live.png"), live_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, "initial_live.png"), live_visualized)

    cv2.destroyAllWindows()


def visualize_final_fields(canonical_field, live_field, view_scaling_factor):
    cv2.imshow("live SDF", sdf_field_to_image(live_field, scale=view_scaling_factor))
    process_cv_esc()
    cv2.imshow("canonical SDF", sdf_field_to_image(canonical_field, scale=view_scaling_factor))
    process_cv_esc()
    cv2.destroyAllWindows()


def save_final_fields(canonical_field, live_field, out_path, view_scaling_factor):
    final_live = sdf_field_to_image(live_field, scale=view_scaling_factor)
    cv2.imwrite(os.path.join(out_path, 'final_live.png'), final_live)
    final_canonical = sdf_field_to_image(canonical_field, scale=view_scaling_factor)
    cv2.imwrite(os.path.join(out_path, "final_canonical.png"), final_canonical)


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


def perform_multiple_tests(start_from_sample=0, data_term_method=DataTermMethod.BASIC,
                           out_path="out2D/Snoopy MultiTest"):
    save_initial_and_final_fields = True
    field_size = 128
    default_value = 1

    offset = [-64, -64, 128]
    line_range = (214, 400)
    frames = list(range(0, 715, 5))
    line_set = line_range[0] + ((line_range[1] - line_range[0]) * np.random.rand(len(frames))).astype(np.int32)
    view_scaling_factor = 1024 // field_size

    calibration_path = "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt"
    if start_from_sample == 0 and os.path.exists(os.path.join(out_path, "output_log.txt")):
        os.unlink(os.path.join(out_path, "output_log.txt"))

    i_sample = 0
    for canonical_frame_index, pixel_row_index in zip(frames, line_set):
        if i_sample < start_from_sample:
            i_sample += 1
            continue
        live_frame_index = canonical_frame_index + 1
        canonical_frame_path = \
            "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_{:0>6d}.png".format(
                canonical_frame_index)
        live_frame_path = \
            "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_{:0>6d}.png".format(
                live_frame_index)
        out_subpath = os.path.join(out_path, "snoopy frames {:0>6d}-{:0>6d} line {:0>3d}"
                                   .format(canonical_frame_index, live_frame_index, pixel_row_index))
        dataset = Dataset(calibration_path, canonical_frame_path, live_frame_path, pixel_row_index, field_size, offset)
        live_field, canonical_field = dataset.generate_2d_sdf_fields(default_value)

        warp_field = np.zeros((field_size, field_size, 2), dtype=np.float32)

        if save_initial_and_final_fields:
            save_initial_fields(canonical_field, live_field, out_subpath, view_scaling_factor)

        print("{:s} OPTIMIZATION BETWEEN FRAMES {:0>6d} AND {:0>6d} ON LINE {:0>3d}{:s}"
              .format(BOLD_LIGHT_CYAN, canonical_frame_index, live_frame_index, pixel_row_index, RESET))

        optimizer = Optimizer2D(out_path=out_subpath,
                                field_size=field_size,
                                default_value=default_value,

                                level_set_term_enabled=False,
                                sobolev_smoothing_enabled=True,

                                data_term_method=data_term_method,
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

        with open(os.path.join(out_path, "output_log.txt"), "a") as output_log:
            output_log.write(
                "Finished optimizing frames {:0>6d}-{:0>6d}, pixel row {:0>3d} in {:d} iterations\n".format(
                    canonical_frame_index, live_frame_index, pixel_row_index, optimizer.last_run_iteration_count))

        if save_initial_and_final_fields:
            save_final_fields(canonical_field, live_field, out_subpath, view_scaling_factor)

        del optimizer
        plt.close('all')
        gc.collect()
        i_sample += 1


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
