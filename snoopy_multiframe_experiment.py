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
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# local
from data_term import DataTermMethod
from dataset import ImageBasedDataset
from smoothing_term import SmoothingTermMethod
from optimizer2d import Optimizer2d, AdaptiveLearningRateMethod, ComputeMethod
from sobolev_filter import generate_1d_sobolev_kernel
from utils.printing import *
from utils.vizualization import save_initial_fields, save_final_fields

import level_set_fusion_optimization as cpp_module


class OptimizerChoice(Enum):
    PYTHON_DIRECT = 0
    PYTHON_VECTORIZED = 1
    CPP = 3


def build_optimizer(optimizer_choice, out_path, field_size, view_scaling_factor=8, max_iterations=100,
                    enable_warp_statistics_logging=False):
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

                                data_term_method=DataTermMethod.BASIC,
                                smoothing_term_method=SmoothingTermMethod.TIKHONOV,
                                adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                                data_term_weight=1.0,
                                smoothing_term_weight=0.2,
                                isomorphic_enforcement_factor=0.1,
                                level_set_term_weight=0.2,

                                maximum_warp_length_lower_threshold=0.1,
                                max_iterations=max_iterations,
                                min_iterations=5,

                                sobolev_kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),

                                enable_component_fields=True,
                                view_scaling_factor=view_scaling_factor)
    elif optimizer_choice == OptimizerChoice.CPP:

        shared_parameters = cpp_module.SharedParameters.get_instance()
        shared_parameters.maximum_iteration_count = max_iterations
        shared_parameters.minimum_iteration_count = 5
        shared_parameters.maximum_warp_length_lower_threshold = 0.1
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
                convergence_status.iteration_limit_reached,
                convergence_status.largest_warp_below_minimum_threshold,
                convergence_status.largest_warp_above_maximum_threshold])


def plot_warp_statistics(out_path, warp_statistics):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # fig, ax1 = plt.subplots(figsize=(10.2, 4.8))
    print(repr(warp_statistics))


def record_convergence_status_log(log, file_path):
    df = pd.DataFrame(log, columns=["canonical frame index", "live frame index", "pixel row index",
                                    "iteration count",
                                    "maximum warp length",
                                    "iteration_limit_reached",
                                    "largest warp below minimum thershold",
                                    "largest warp above minimum threshold"])
    df.to_csv(file_path)


def perform_multiple_tests(start_from_sample=0, data_term_method=DataTermMethod.BASIC,
                           out_path="out2D/Snoopy MultiTest", input_case_file=None,
                           calibration_path=
                           "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
                           frame_path=
                           "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/"):
    # CANDIDATES FOR ARGS
    save_initial_and_final_fields = False
    optimizer_choice = OptimizerChoice.CPP
    enable_warp_statistics_logging = True

    field_size = 128
    default_value = 1
    rebuild_optimizer = OptimizerChoice != OptimizerChoice.CPP

    offset = [-64, -64, 128]
    line_range = (214, 400)

    if input_case_file:
        frame_and_row_set = np.genfromtxt(input_case_file, delimiter=",", dtype=np.int32)
        canonical_frame_and_row_set = np.concatenate(
            (frame_and_row_set[:, 0].reshape(-1, 1), frame_and_row_set[:, 2].reshape(-1, 1)), axis=1)
    else:
        frame_set = list(range(0, 715, 5))
        pixel_row_set = line_range[0] + ((line_range[1] - line_range[0]) * np.random.rand(len(frame_set))).astype(
            np.int32)
        canonical_frame_and_row_set = zip(frame_set, pixel_row_set)

    view_scaling_factor = 1024 // field_size

    convergence_status_log = []
    convergence_status_log_file_path = os.path.join(out_path, "convergence_status_log.csv")

    save_log_every_n_runs = 5

    # dataset location
    frame_path_format_string = frame_path + os.path.sep + "depth_{:0>6d}.png";

    if start_from_sample == 0 and os.path.exists(os.path.join(out_path, "output_log.txt")):
        os.unlink(os.path.join(out_path, "output_log.txt"))

    i_sample = 0

    optimizer = None if rebuild_optimizer else \
        build_optimizer(optimizer_choice, out_path, field_size, view_scaling_factor=8, max_iterations=100,
                        enable_warp_statistics_logging=enable_warp_statistics_logging)

    for canonical_frame_index, pixel_row_index in canonical_frame_and_row_set:
        if i_sample < start_from_sample:
            i_sample += 1
            continue
        live_frame_index = canonical_frame_index + 1

        canonical_frame_path = frame_path_format_string.format(canonical_frame_index)
        live_frame_path = frame_path_format_string.format(live_frame_index)
        out_subpath = os.path.join(out_path, "snoopy frames {:0>6d}-{:0>6d} line {:0>3d}"
                                   .format(canonical_frame_index, live_frame_index, pixel_row_index))
        dataset = ImageBasedDataset(calibration_path, canonical_frame_path, live_frame_path, pixel_row_index,
                                    field_size, offset)
        live_field, canonical_field = dataset.generate_2d_sdf_fields(default_value)

        if save_initial_and_final_fields:
            save_initial_fields(canonical_field, live_field, out_subpath, view_scaling_factor)

        print("{:s} OPTIMIZATION BETWEEN FRAMES {:0>6d} AND {:0>6d} ON LINE {:0>3d}{:s}"
              .format(BOLD_LIGHT_CYAN, canonical_frame_index, live_frame_index, pixel_row_index, RESET))

        if rebuild_optimizer:
            optimizer = build_optimizer(optimizer_choice, out_subpath, field_size, view_scaling_factor=8,
                                        max_iterations=400,
                                        enable_warp_statistics_logging=enable_warp_statistics_logging)

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
                plot_warp_statistics(out_subpath, warp_statistics)

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
