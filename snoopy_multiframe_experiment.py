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

# libraries
import numpy as np
from matplotlib import pyplot as plt

# local
from data_term import DataTermMethod
from dataset import Dataset
from smoothing_term import SmoothingTermMethod
from optimizer2d import Optimizer2D, AdaptiveLearningRateMethod
from sobolev_filter import generate_1d_sobolev_kernel
from printing import *
from vizualization import save_initial_fields, save_final_fields


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
