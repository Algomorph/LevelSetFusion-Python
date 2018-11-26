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
# libraries
import numpy as np

from data_term import DataTermMethod
from dataset import datasets, DataToUse
from smoothing_term import SmoothingTermMethod
from tsdf_field_generation import generate_initial_orthographic_2d_tsdf_fields
from optimizer2d import Optimizer2d, AdaptiveLearningRateMethod, ComputeMethod
from sobolev_filter import generate_1d_sobolev_kernel
from utils.visualization import visualize_and_save_initial_fields, visualize_final_fields


def perform_single_test():
    visualize_and_save_initial_and_final_fields = True
    field_size = 128
    default_value = 1
    out_path = "output/out2D"
    data_to_use = DataToUse.SYNTHETIC3D_SUZANNE_TWIST

    if data_to_use == DataToUse.GENEREATED2D:
        live_field, canonical_field = \
            generate_initial_orthographic_2d_tsdf_fields(field_size=field_size,
                                                         live_smoothing_kernel_size=0,
                                                         canonical_smoothing_kernel_size=0,
                                                         default_value=default_value)
    else:
        live_field, canonical_field = datasets[data_to_use].generate_2d_sdf_fields()
        field_size = datasets[data_to_use].field_size

    warp_field = np.zeros((field_size, field_size, 2), dtype=np.float32)
    view_scaling_factor = 1024 // field_size

    if visualize_and_save_initial_and_final_fields:
        visualize_and_save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor)

    optimizer = Optimizer2d(out_path=out_path,
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
                            max_iterations=5,

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
