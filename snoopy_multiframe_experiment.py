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

# stdlib
import os
import os.path
import gc
from enum import Enum

# libraries
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# local
from data_term import DataTermMethod
from dataset import ImageBasedDataset, MaskedImageBasedDataset
from smoothing_term import SmoothingTermMethod
from optimizer2d import Optimizer2d, AdaptiveLearningRateMethod, ComputeMethod
from sobolev_filter import generate_1d_sobolev_kernel
from utils.printing import *
from utils.visualization import save_initial_fields, save_final_fields, rescale_depth_to_8bit, highlight_row_on_gray
import level_set_fusion_optimization as cpp_module
import utils.sampling as sampling


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
        optimizer = Optimizer2d(out_path=out_path,
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


def perform_multiple_tests(start_from_sample=0, data_term_method=DataTermMethod.BASIC,
                           optimizer_choice=OptimizerChoice.CPP,
                           out_path="out2D/Snoopy MultiTest", input_case_file=None,
                           calibration_path=
                           "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
                           frame_path=
                           "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/"):
    # CANDIDATES FOR ARGS
    save_initial_and_final_fields = input_case_file is not None
    enable_warp_statistics_logging = input_case_file is not None
    save_frame_images = input_case_file is not None
    use_masks = True
    check_empty_row = True

    field_size = 128

    rebuild_optimizer = optimizer_choice != OptimizerChoice.CPP
    max_iterations = 400 if optimizer_choice == OptimizerChoice.CPP else 100

    # dataset location
    frame_path_format_string = frame_path + os.path.sep + "depth_{:0>6d}.png"
    mask_path_format_string = frame_path + os.path.sep + "mask_{:0>6d}.png"

    # region ================ Generation of lists of frames & pixel rows to work with ==================================

    offset = [-64, -64, 128]
    line_range = (214, 400)

    if input_case_file:
        frame_row_and_focus_set = np.genfromtxt(input_case_file, delimiter=",", dtype=np.int32)
        # drop column headers
        frame_row_and_focus_set = frame_row_and_focus_set[1:]
        # drop live frame indexes
        frame_row_and_focus_set = np.concatenate(
            (frame_row_and_focus_set[:, 0].reshape(-1, 1), frame_row_and_focus_set[:, 2].reshape(-1, 1),
             frame_row_and_focus_set[:, 3:]), axis=1)
    else:
        frame_set = list(range(0, 715, 5))
        pixel_row_set = line_range[0] + ((line_range[1] - line_range[0]) * np.random.rand(len(frame_set))).astype(
            np.int32)
        focus_coordinates = np.array([0, 0] * len(frame_set))
        frame_row_and_focus_set = zip(frame_set, pixel_row_set, focus_coordinates)
        if check_empty_row:
            # replace empty rows
            new_pixel_row_set = []
            for canonical_frame_index, pixel_row_index in frame_row_and_focus_set:
                live_frame_index = canonical_frame_index + 1
                canonical_frame_path = frame_path_format_string.format(canonical_frame_index)
                canonical_mask_path = mask_path_format_string.format(canonical_frame_index)
                live_frame_path = frame_path_format_string.format(live_frame_index)
                live_mask_path = mask_path_format_string.format(live_frame_index)
                while is_image_row_empty(canonical_frame_path, canonical_mask_path, pixel_row_index, use_masks) or \
                        is_image_row_empty(live_frame_path, live_mask_path, pixel_row_index, use_masks):
                    pixel_row_index = line_range[0] + (line_range[1] - line_range[0]) * np.random.rand()
                new_pixel_row_set.append(pixel_row_index)
            frame_row_and_focus_set = zip(frame_set, pixel_row_set, focus_coordinates)

    view_scaling_factor = 1024 // field_size

    # endregion ========================================================================================================

    # logging
    convergence_status_log = []
    convergence_status_log_file_path = os.path.join(out_path, "convergence_status_log.csv")

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
        out_subpath = os.path.join(out_path, "snoopy frames {:0>6d}-{:0>6d} line {:0>3d}"
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
            dataset = MaskedImageBasedDataset(calibration_path, canonical_frame_path, canonical_mask_path,
                                              live_frame_path, live_mask_path, pixel_row_index,
                                              field_size, offset)
        else:
            dataset = ImageBasedDataset(calibration_path, canonical_frame_path, live_frame_path, pixel_row_index,
                                        field_size, offset)

        live_field, canonical_field = dataset.generate_2d_sdf_fields()

        if save_initial_and_final_fields:
            save_initial_fields(canonical_field, live_field, out_subpath, view_scaling_factor)

        print("{:s} OPTIMIZATION BETWEEN FRAMES {:0>6d} AND {:0>6d} ON LINE {:0>3d}{:s}"
              .format(BOLD_LIGHT_CYAN, canonical_frame_index, live_frame_index, pixel_row_index, RESET), end="")

        if rebuild_optimizer:
            optimizer = build_optimizer(optimizer_choice, out_subpath, field_size, view_scaling_factor=8,
                                        max_iterations=max_iterations,
                                        enable_warp_statistics_logging=enable_warp_statistics_logging,
                                        data_term_method=data_term_method)

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

        converged = not optimizer.get_convergence_status().iteration_limit_reached
        if converged:
            print(": CONVERGED")
        else:
            print(": NOT CONVERGED")

        log_convergence_status(convergence_status_log, optimizer.get_convergence_status(),
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
