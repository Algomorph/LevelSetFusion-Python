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
import os
import os.path
# libraries
import progressbar
# local
import utils.visualization as viz
import nonrigid_opt.hierarchical.hierarchical_optimization_visualizer as viz_ho
from experiment import dataset as ds
from tsdf import generation as tsdf
from nonrigid_opt import field_warping as resampling
import nonrigid_opt.slavcheva.sobolev_filter as sob
import utils.sampling as sampling
# has to be compiled and included in PYTHONPATH first
import level_set_fusion_optimization as cpp

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def print_convergence_reports(reports):
    for i_level, report in enumerate(reports):
        print("[LEVEL", i_level, "]")
        print(report)


def main():
    data_to_use = ds.PredefinedDatasetEnum.REAL3D_SNOOPY_SET05
    tsdf_generation_method = cpp.tsdf.FilteringMethod.NONE

    out_path = "output/ho3d/single"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    generate_test_data = False

    live_field, canonical_field = \
        ds.datasets[data_to_use].generate_3d_sdf_fields(method=tsdf_generation_method, smoothing_coefficient=0.5)

    view_scaling_factor = 1024 // ds.datasets[data_to_use].field_size

    if generate_test_data:
        live_field = live_field[36:52, 21:37].copy()
        canonical_field = canonical_field[36:52, 21:37].copy()

    maximum_warp_update_threshold = 0.01
    maximum_iteration_count = 100

    verbosity_parameters_cpp = cpp.HierarchicalOptimizer3d.VerbosityParameters(
        print_max_warp_update=True,
        print_iteration_mean_tsdf_difference=True,
        print_iteration_std_tsdf_difference=True,
        print_iteration_data_energy=True,
        print_iteration_tikhonov_energy=True,
    )

    logging_parameters_cpp = cpp.HierarchicalOptimizer3d.LoggingParameters(
        collect_per_level_convergence_reports=True,
        collect_per_level_iteration_data=False
    )
    resampling_strategy_cpp = cpp.HierarchicalOptimizer3d.ResamplingStrategy.NEAREST_AND_AVERAGE
    # resampling_strategy_cpp = ho_cpp.HierarchicalOptimizer3d.ResamplingStrategy.LINEAR

    optimizer = cpp.HierarchicalOptimizer3d(
        tikhonov_term_enabled=False,
        gradient_kernel_enabled=False,

        maximum_chunk_size=8,
        rate=0.1,
        maximum_iteration_count=maximum_iteration_count,
        maximum_warp_update_threshold=maximum_warp_update_threshold,

        data_term_amplifier=1.0,
        tikhonov_strength=0.0,
        kernel=sob.generate_1d_sobolev_kernel(size=7, strength=0.1),

        resampling_strategy=resampling_strategy_cpp,

        verbosity_parameters=verbosity_parameters_cpp,
        logging_parameters=logging_parameters_cpp
    )

    warp_field = optimizer.optimize(canonical_field, live_field)
    print("Warp [min, mean, max]:", warp_field.min(), warp_field.mean(), warp_field.max())

    print("===================================================================================")
    print_convergence_reports(optimizer.get_per_level_convergence_reports())
    # telemetry_log = optimizer.get_per_level_iteration_data()
    # metadata = viz_ho.get_telemetry_metadata(telemetry_log)
    # frame_count = viz_ho.get_number_of_frames_to_save_from_telemetry_logs([telemetry_log])
    # progress_bar = progressbar.ProgressBar(max_value=frame_count)
    # viz_ho.convert_cpp_telemetry_logs_to_video(telemetry_log, metadata, canonical_field, live_field, out_path,
    #                                            progress_bar=progress_bar)

    # warped_live = resampling.warp_field(live_field, warp_field)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
