#  ================================================================
#  Created by Gregory Kramida on 11/26/18.
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

# contains code for running a single experiment on a specific frame of some dataset

# stdlib
import time
import os
# libraries
import numpy as np
# local
from data_term import DataTermMethod
from dataset import datasets, DataToUse, MaskedImageBasedSingleFrameDataset, ImageBasedSingleFrameDataset
from smoothing_term import SmoothingTermMethod
from tsdf_field_generation import generate_initial_orthographic_2d_tsdf_fields, DepthInterpolationMethod
from slavcheva_optimizer2d import SlavchevaOptimizer2d, AdaptiveLearningRateMethod, ComputeMethod
from sobolev_filter import generate_1d_sobolev_kernel
from utils.visualization import visualize_and_save_initial_fields, visualize_final_fields
import experiment_shared_routines as shared


def perform_single_test(depth_interpolation_method=DepthInterpolationMethod.NONE, out_path="output/out2D",
                        frame_path="", calibration_path="calib.txt", canonical_frame_index=-1, pixel_row_index=-1,
                        z_offset=128, draw_tsdfs_and_exit=False):
    visualize_and_save_initial_and_final_fields = True
    field_size = 128
    default_value = 1

    if pixel_row_index < 0 and canonical_frame_index < 0:
        data_to_use = DataToUse.REAL3D_SNOOPY_SET04

        if data_to_use == DataToUse.GENEREATED2D:
            live_field, canonical_field = \
                generate_initial_orthographic_2d_tsdf_fields(field_size=field_size,
                                                             live_smoothing_kernel_size=0,
                                                             canonical_smoothing_kernel_size=0,
                                                             default_value=default_value)
        else:
            live_field, canonical_field = \
                datasets[data_to_use].generate_2d_sdf_fields(method=depth_interpolation_method)
            field_size = datasets[data_to_use].field_size
    else:
        frame_count, frame_filename_format, use_masks = shared.check_frame_count_and_format(frame_path)
        if frame_filename_format == shared.FrameFilenameFormat.SIX_DIGIT:
            frame_path_format_string = frame_path + os.path.sep + "depth_{:0>6d}.png"
            mask_path_format_string = frame_path + os.path.sep + "mask_{:0>6d}.png"
        else:  # has to be FIVE_DIGIT
            frame_path_format_string = frame_path + os.path.sep + "depth_{:0>5d}.png"
            mask_path_format_string = frame_path + os.path.sep + "mask_{:0>5d}.png"
        live_frame_index = canonical_frame_index + 1
        canonical_frame_path = frame_path_format_string.format(canonical_frame_index)
        canonical_mask_path = mask_path_format_string.format(canonical_frame_index)
        live_frame_path = frame_path_format_string.format(live_frame_index)
        live_mask_path = mask_path_format_string.format(live_frame_index)

        offset = [-64, -64, z_offset]
        # Generate SDF fields
        if use_masks:
            dataset = MaskedImageBasedSingleFrameDataset(calibration_path, canonical_frame_path, canonical_mask_path,
                                                         live_frame_path, live_mask_path, pixel_row_index,
                                                         field_size, offset)
        else:
            dataset = ImageBasedSingleFrameDataset(calibration_path, canonical_frame_path, live_frame_path,
                                                   pixel_row_index, field_size, offset)

        live_field, canonical_field = dataset.generate_2d_sdf_fields(method=depth_interpolation_method)

    warp_field = np.zeros((field_size, field_size, 2), dtype=np.float32)
    view_scaling_factor = 1024 // field_size

    if visualize_and_save_initial_and_final_fields:
        visualize_and_save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor)

    if draw_tsdfs_and_exit:
        return

    optimizer = SlavchevaOptimizer2d(out_path=out_path,
                                     field_size=field_size,
                                     default_value=default_value,

                                     compute_method=ComputeMethod.VECTORIZED,

                                     level_set_term_enabled=False,
                                     sobolev_smoothing_enabled=True,

                                     data_term_method=DataTermMethod.BASIC,
                                     smoothing_term_method=SmoothingTermMethod.TIKHONOV,
                                     adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                                     data_term_weight=1.0,
                                     smoothing_term_weight=0.2,
                                     isomorphic_enforcement_factor=0.1,
                                     level_set_term_weight=0.2,

                                     maximum_warp_length_lower_threshold=0.05,
                                     max_iterations=100,

                                     sobolev_kernel=generate_1d_sobolev_kernel(size=7 if field_size > 7 else 3, strength=0.1),

                                     enable_component_fields=True,
                                     view_scaling_factor=view_scaling_factor)

    start_time = time.time()
    optimizer.optimize(live_field, canonical_field)
    end_time = time.time()
    print("Total optimization runtime: {:f}".format(end_time - start_time))
    optimizer.plot_logged_sdf_and_warp_magnitudes()
    optimizer.plot_logged_energies_and_max_warps()

    if visualize_and_save_initial_and_final_fields:
        visualize_final_fields(canonical_field, live_field, view_scaling_factor)
