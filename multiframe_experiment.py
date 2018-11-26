#  ================================================================
#  Created by Gregory Kramida on 10/10/18.
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

# contains code for experimenting on multiple frames / cases and logging/recording corresponding results

# stdlib
import os
import os.path
import gc
from enum import Enum
import re

# libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# local
from build_optimizer import OptimizerChoice, build_optimizer
from data_term import DataTermMethod
from dataset import ImageBasedSingleFrameDataset, MaskedImageBasedSingleFrameDataset
from tsdf_field_generation import DepthInterpolationMethod
from utils.printing import *
from utils.visualization import save_initial_fields, save_final_fields, rescale_depth_to_8bit, highlight_row_on_gray
import utils.sampling as sampling


def log_convergence_status(log, convergence_status, canonical_frame_index, live_frame_index, pixel_row_index):
    log.append([canonical_frame_index,
                live_frame_index,
                pixel_row_index,
                convergence_status.iteration_count,
                convergence_status.max_warp_length,
                convergence_status.max_warp_location.x,
                convergence_status.max_warp_location.y,
                convergence_status.iteration_limit_reached,
                convergence_status.largest_warp_below_minimum_threshold,
                convergence_status.largest_warp_above_maximum_threshold])


def record_convergence_status_log(log, file_path):
    df = pd.DataFrame(log, columns=["canonical frame index", "live frame index", "pixel row index",
                                    "iteration count",
                                    "max warp length",
                                    "max warp x",
                                    "max warp y",
                                    "iteration limit reached",
                                    "largest warp below minimum threshold",
                                    "largest warp above maximum threshold"])
    df.to_csv(file_path)


def record_cases_files(log, out_directory):
    df = pd.DataFrame(log, columns=["canonical frame index", "live frame index", "pixel row index",
                                    "iteration count",
                                    "max warp length",
                                    "max warp x",
                                    "max warp y",
                                    "iteration limit reached",
                                    "largest warp below minimum threshold",
                                    "largest warp above maximum threshold"])
    cases_df = df.drop(columns=["iteration count", "max warp length", "iteration limit reached",
                                "largest warp below minimum threshold", "largest warp above maximum threshold"])

    bad_cases_df = cases_df[df["iteration limit reached"] == True]  # TODO: PyCharm bug
    good_cases_df = cases_df[df["iteration limit reached"] == False]

    cases_df.to_csv(os.path.join(out_directory, "cases.csv"), index=False)
    bad_cases_df.to_csv(os.path.join(out_directory, "bad_cases.csv"), index=False)
    good_cases_df.to_csv(os.path.join(out_directory, "good_cases.csv"), index=False)


def plot_warp_statistics(out_path, warp_statistics, convergence_threshold=0.1, extra_path=None):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # C++ definition of the struct underlying each row in warp_statistics (careful, may be outdated!):
    # float ratio_of_warps_above_minimum_threshold = 0.0;
    # float max_warp_length = 0.0
    # float mean_warp_length = 0.0;
    # float standard_deviation_of_warp_length = 0.0;
    ratios_of_warps_above_minimum_threshold = warp_statistics[:, 0]
    maximum_warp_lengths = warp_statistics[:, 1]
    mean_warp_lengths = warp_statistics[:, 2]
    standard_deviations_of_warp_lengths = warp_statistics[:, 3]
    convergence_threshold_marks = np.array([convergence_threshold] * len(mean_warp_lengths))

    color = "tab:red"
    dpi = 96
    fig, ax_ratios = plt.subplots(figsize=(3000 / dpi, 1000 / dpi), dpi=dpi)
    ax_ratios.set_xlabel("iteration number")
    ax_ratios.set_ylabel("Ratio of warp lengths below convergence threshold", color=color)
    ax_ratios.plot(ratios_of_warps_above_minimum_threshold * 100, color=color, label="% of warp lengths above "
                                                                                     "convergence threshold")
    ax_ratios.tick_params(axis='y', labelcolor=color)
    ax_ratios.legend(loc='upper left')

    color = "tab:blue"
    ax_lengths = ax_ratios.twinx()
    ax_lengths.set_ylabel("warp_length", color=color)
    ax_lengths.plot(maximum_warp_lengths, "c-", label="maximum warp length")
    ax_lengths.plot(mean_warp_lengths, "b-", label="mean warp length")
    ax_lengths.plot(mean_warp_lengths + standard_deviations_of_warp_lengths, "g-",
                    label="standard deviation of warp length")
    ax_lengths.plot(mean_warp_lengths - standard_deviations_of_warp_lengths, "g-")
    ax_lengths.plot(convergence_threshold_marks, "k-", label="convergence threshold")
    ax_lengths.plot(convergence_threshold)
    ax_lengths.plot()
    ax_lengths.tick_params(axis='y', labelcolor=color)
    ax_lengths.legend(loc='upper right')

    fig.tight_layout()
    if extra_path:
        plt.savefig(extra_path)
    plt.savefig(os.path.join(out_path, "warp_statistics.png"))
    plt.close(fig)


def is_unmasked_image_row_empty(path, ix_row):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return np.sum(image[ix_row]) == 0


def is_masked_image_row_empty(image_path, mask_path, ix_row):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    image[mask == 0] = 0
    return np.sum(image[ix_row]) == 0


def is_image_row_empty(image_path, mask_path, ix_row, check_masked):
    if check_masked:
        return is_masked_image_row_empty(image_path, mask_path, ix_row)
    else:
        return is_unmasked_image_row_empty(image_path, ix_row)


class FrameFilenameFormat(Enum):
    FIVE_DIGIT = 0
    SIX_DIGIT = 1


def check_frame_count_and_format(frames_path):
    five_digit_pattern = re.compile(r"^depth\_\d{5}[.]png$")
    six_digit_pattern = re.compile(r"^depth\_\d{6}[.]png$")
    five_digit_counter = 0
    six_digit_counter = 0
    for filename in os.listdir(frames_path):
        if re.findall(five_digit_pattern, filename):
            five_digit_counter += 1
        elif re.findall(six_digit_pattern, filename):
            six_digit_counter += 1
    frame_count = max(five_digit_counter, six_digit_counter)
    format = FrameFilenameFormat.FIVE_DIGIT if five_digit_counter > six_digit_counter else FrameFilenameFormat.SIX_DIGIT
    return frame_count, format


def perform_multiple_tests(start_from_sample=0, data_term_method=DataTermMethod.BASIC,
                           optimizer_choice=OptimizerChoice.CPP,
                           out_path="out2D/Snoopy MultiTest", input_case_file=None,
                           calibration_path=
                           "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
                           frame_path=
                           "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/",
                           z_offset=128,
                           depth_interpolation_method=DepthInterpolationMethod.NONE):
    # CANDIDATES FOR ARGS

    save_initial_and_final_fields = input_case_file is not None
    enable_warp_statistics_logging = input_case_file is not None
    save_frame_images = input_case_file is not None
    use_masks = True
    check_empty_row = True
    # TODO a tiled image with 6x6 bad cases and 6x6 good cases (SDF fields)
    save_tiled_good_vs_bad_case_comparison_image = True

    rebuild_optimizer = optimizer_choice != OptimizerChoice.CPP
    max_iterations = 400 if optimizer_choice == OptimizerChoice.CPP else 100

    # dataset location
    frame_count, frame_filename_format = check_frame_count_and_format(frame_path)
    if frame_filename_format == FrameFilenameFormat.SIX_DIGIT:
        frame_path_format_string = frame_path + os.path.sep + "depth_{:0>6d}.png"
        mask_path_format_string = frame_path + os.path.sep + "mask_{:0>6d}.png"
    else:  # has to be FIVE_DIGIT
        frame_path_format_string = frame_path + os.path.sep + "depth_{:0>5d}.png"
        mask_path_format_string = frame_path + os.path.sep + "mask_{:0>5d}.png"

    # region ================ Generation of lists of frames & pixel rows to work with ==================================

    # CANDIDATES FOR ARGS
    field_size = 128
    offset = [-64, -64, z_offset]
    line_range = (214, 400)

    if input_case_file:
        frame_row_and_focus_set = np.genfromtxt(input_case_file, delimiter=",", dtype=np.int32)
        # drop column headers
        frame_row_and_focus_set = frame_row_and_focus_set[1:]
        # drop live frame indexes
        frame_row_and_focus_set = np.concatenate(
            (frame_row_and_focus_set[:, 0].reshape(-1, 1), frame_row_and_focus_set[:, 2].reshape(-1, 1),
             frame_row_and_focus_set[:, 3:5]), axis=1)
    else:
        frame_set = list(range(0, frame_count - 1, 5))
        pixel_row_set = line_range[0] + ((line_range[1] - line_range[0]) * np.random.rand(len(frame_set))).astype(
            np.int32)
        focus_x = np.zeros((len(frame_set), 1,))
        focus_y = np.zeros((len(frame_set), 1,))
        frame_row_and_focus_set = zip(frame_set, pixel_row_set, focus_x, focus_y)
        if check_empty_row:
            # replace empty rows
            new_pixel_row_set = []
            for canonical_frame_index, pixel_row_index, _, _ in frame_row_and_focus_set:
                live_frame_index = canonical_frame_index + 1
                canonical_frame_path = frame_path_format_string.format(canonical_frame_index)
                canonical_mask_path = mask_path_format_string.format(canonical_frame_index)
                live_frame_path = frame_path_format_string.format(live_frame_index)
                live_mask_path = mask_path_format_string.format(live_frame_index)
                while is_image_row_empty(canonical_frame_path, canonical_mask_path, pixel_row_index, use_masks) or \
                        is_image_row_empty(live_frame_path, live_mask_path, pixel_row_index, use_masks):
                    pixel_row_index = line_range[0] + (line_range[1] - line_range[0]) * np.random.rand()
                new_pixel_row_set.append(pixel_row_index)
            frame_row_and_focus_set = zip(frame_set, pixel_row_set, focus_x, focus_y)

    view_scaling_factor = 1024 // field_size

    # endregion ========================================================================================================

    # logging
    convergence_status_log = []
    convergence_status_log_file_path = os.path.join(out_path, "convergence_status_log.csv")

    max_case_count = 36
    good_case_sdfs = []
    bad_case_sdfs = []

    save_log_every_n_runs = 5

    if start_from_sample == 0 and os.path.exists(os.path.join(out_path, "output_log.txt")):
        os.unlink(os.path.join(out_path, "output_log.txt"))

    i_sample = 0

    optimizer = None if rebuild_optimizer else \
        build_optimizer(optimizer_choice, out_path, field_size, view_scaling_factor=8, max_iterations=max_iterations,
                        enable_warp_statistics_logging=enable_warp_statistics_logging,
                        data_term_method=data_term_method)

    # run the optimizers
    for canonical_frame_index, pixel_row_index, focus_x, focus_y in frame_row_and_focus_set:
        if i_sample < start_from_sample:
            i_sample += 1
            continue

        sampling.set_focus_coordinates(focus_x, focus_y)

        live_frame_index = canonical_frame_index + 1
        out_subpath = os.path.join(out_path, "frames {:0>6d}-{:0>6d} line {:0>3d}"
                                   .format(canonical_frame_index, live_frame_index, pixel_row_index))

        canonical_frame_path = frame_path_format_string.format(canonical_frame_index)
        canonical_mask_path = mask_path_format_string.format(canonical_frame_index)
        live_frame_path = frame_path_format_string.format(live_frame_index)
        live_mask_path = mask_path_format_string.format(live_frame_index)

        if save_frame_images:
            def highlight_row_and_save_image(path_to_original, output_name, ix_row):
                output_image = highlight_row_on_gray(
                    rescale_depth_to_8bit(cv2.imread(path_to_original, cv2.IMREAD_UNCHANGED)), ix_row)
                cv2.imwrite(os.path.join(out_subpath, output_name), output_image)

            highlight_row_and_save_image(canonical_frame_path, "canonical_frame_rh.png", pixel_row_index)
            highlight_row_and_save_image(live_frame_path, "live_frame_rh.png", pixel_row_index)

        # Generate SDF fields
        if use_masks:
            dataset = MaskedImageBasedSingleFrameDataset(calibration_path, canonical_frame_path, canonical_mask_path,
                                                         live_frame_path, live_mask_path, pixel_row_index,
                                                         field_size, offset)
        else:
            dataset = ImageBasedSingleFrameDataset(calibration_path, canonical_frame_path, live_frame_path,
                                                   pixel_row_index, field_size, offset)

        live_field, canonical_field = dataset.generate_2d_sdf_fields(method=depth_interpolation_method)

        if save_initial_and_final_fields:
            save_initial_fields(canonical_field, live_field, out_subpath, view_scaling_factor)

        print("{:s} OPTIMIZATION BETWEEN FRAMES {:0>6d} AND {:0>6d} ON LINE {:0>3d}{:s}"
              .format(BOLD_LIGHT_CYAN, canonical_frame_index, live_frame_index, pixel_row_index, RESET), end="")

        if rebuild_optimizer:
            optimizer = build_optimizer(optimizer_choice, out_subpath, field_size, view_scaling_factor=8,
                                        max_iterations=max_iterations,
                                        enable_warp_statistics_logging=enable_warp_statistics_logging,
                                        data_term_method=data_term_method)
        original_live_field = live_field.copy()
        live_field = optimizer.optimize(live_field, canonical_field)

        # ===================== LOG AFTER-RUN RESULTS ==================================================================

        if optimizer_choice != OptimizerChoice.CPP:
            # call python-specific logging routines
            optimizer.plot_logged_sdf_and_warp_magnitudes()
            optimizer.plot_logged_energies_and_max_warps()
        else:
            # call C++-specific logging routines
            if enable_warp_statistics_logging:
                warp_statistics = optimizer.get_warp_statistics_as_matrix()
                root_subpath = os.path.join(out_path, "warp_statistics_frames_{:0>6d}-{:0>6d}_row_{:0>3d}.png"
                                            .format(canonical_frame_index, live_frame_index, pixel_row_index))
                plot_warp_statistics(out_subpath, warp_statistics, extra_path=root_subpath)

        convergence_status = optimizer.get_convergence_status()
        if not convergence_status.iteration_limit_reached:
            if convergence_status.largest_warp_above_maximum_threshold:
                print(": DIVERGED", end="")
            else:
                print(": CONVERGED", end="")

            if (save_tiled_good_vs_bad_case_comparison_image and convergence_status.largest_warp_above_maximum_threshold
                    and len(good_case_sdfs) < max_case_count):
                good_case_sdfs.append((original_live_field, live_field))

        else:
            print(": NOT CONVERGED", end="")
            if save_tiled_good_vs_bad_case_comparison_image and len(bad_case_sdfs) < max_case_count:
                bad_case_sdfs.append((original_live_field, live_field))
        print(" IN", convergence_status.iteration_count, "ITERATIONS")

        log_convergence_status(convergence_status_log, convergence_status,
                               canonical_frame_index, live_frame_index, pixel_row_index)

        if save_initial_and_final_fields:
            save_final_fields(canonical_field, live_field, out_subpath, view_scaling_factor)

        if rebuild_optimizer:
            del optimizer
            plt.close('all')
            gc.collect()

        i_sample += 1

        if i_sample % save_log_every_n_runs == 0:
            record_convergence_status_log(convergence_status_log, convergence_status_log_file_path)

    record_convergence_status_log(convergence_status_log, convergence_status_log_file_path)
    record_cases_files(convergence_status_log, out_path)
    if save_tiled_good_vs_bad_case_comparison_image:
        pass
        # save_tiled_comparison_image(good_case_sdfs, bad_case_sdfs) #TODO