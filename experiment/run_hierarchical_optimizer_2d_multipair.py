#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 12/3/18.
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
import sys

# libraries
import numpy as np

# local
from nonrigid_opt import hns_visualizer as hnsov, hns_optimizer2d as hnso
from experiment import dataset as ds
from tsdf import generation as tsdf
from utils import field_resampling as resampling
import utils.sampling as sampling
from nonrigid_opt.sobolev_filter import generate_1d_sobolev_kernel

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def main():
    data_to_use = ds.PredefinedDatasetEnum.REAL3D_SNOOPY_SET03
    depth_interpolation_method = tsdf.GenerationMethod.NONE
    out_path = "output/hnso"
    sampling.set_focus_coordinates(0, 0)
    generate_test_data = False

    live_field, canonical_field = \
        ds.datasets[data_to_use].generate_2d_sdf_fields(method=depth_interpolation_method)

    if generate_test_data:
        live_field = live_field[36:52, 21:37].copy()
        canonical_field = canonical_field[36:52, 21:37].copy()
        print("initial live:")
        print(repr(live_field))
        print("canonical:")
        print(repr(canonical_field))

    optimizer = hnso.HierarchicalNonrigidSLAMOptimizer2d(
        rate=0.2,
        data_term_amplifier=1.0,
        tikhonov_strength=0.0,
        kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),
        maximum_warp_update_threshold=0.001,
        verbosity_parameters=hnso.HierarchicalNonrigidSLAMOptimizer2d.VerbosityParameters(
            print_max_warp_update=True,
            print_iteration_data_energy=True,
            print_iteration_tikhonov_energy=True,
        ),
        visualization_parameters=hnsov.HNSOVisualizer.Parameters(
            out_path=out_path,
            save_live_progression=True,
            save_initial_fields=True,
            save_final_fields=True,
            save_warp_field_progression=True,
            save_data_gradients=True,
            save_tikhonov_gradients=False
        )
    )
    warp_field = optimizer.optimize(canonical_field, live_field)

    if generate_test_data:
        resampled_live = resampling.resample_field(live_field, warp_field)
        print("final live:")
        print(repr(resampled_live))
        print("warp field:")
        print(repr(warp_field))

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
