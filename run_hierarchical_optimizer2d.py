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
# local
import utils.visualization as viz
from nonrigid_opt import hierarchical_optimization_visualizer as hov, hierarchical_optimizer2d as ho
from experiment import dataset as ds
from tsdf import generation as tsdf
from utils import field_resampling as resampling
import utils.sampling as sampling
from nonrigid_opt.sobolev_filter import generate_1d_sobolev_kernel
# has to be compiled and included in PYTHONPATH first
import level_set_fusion_optimization as cpp_module

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def make_python_optimizer(out_path):
    optimizer = ho.HierarchicalOptimizer2d(
        tikhonov_term_enabled=False,
        gradient_kernel_enabled=False,

        maximum_chunk_size=8,
        rate=0.2,
        maximum_iteration_count=25,
        maximum_warp_update_threshold=0.001,

        data_term_amplifier=1.0,
        tikhonov_strength=0.0,
        kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),

        verbosity_parameters=ho.HierarchicalOptimizer2d.VerbosityParameters(
            print_max_warp_update=True,
            print_iteration_data_energy=True,
            print_iteration_tikhonov_energy=True,
        ),
        visualization_parameters=hov.HNSOVisualizer.Parameters(
            out_path=out_path,
            save_live_progression=True,
            save_initial_fields=True,
            save_final_fields=True,
            save_warp_field_progression=True,
            save_data_gradients=True,
            save_tikhonov_gradients=False
        )
    )
    return optimizer


def make_cpp_optimizer(out_path):
    optimizer = cpp_module.HierarchicalOptimizer2d(
        tikhonov_term_enabled=False,
        gradient_kernel_enabled=False,

        maximum_chunk_size=8,
        rate=0.2,
        maximum_iteration_count=1000,
        maximum_warp_update_threshold=0.001,

        data_term_amplifier=1.0,
        tikhonov_strength=0.0,
        kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),

        verbosity_parameters=cpp_module.HierarchicalOptimizer2d.VerbosityParameters(
            print_max_warp_update=True,
            print_iteration_mean_tsdf_difference=True,
            print_iteration_std_tsdf_difference=True,
            print_iteration_data_energy=True,
            print_iteration_tikhonov_energy=True,
        ),
        logging_parameters=cpp_module.HierarchicalOptimizer2d.LoggingParameters(
            collect_per_level_convergence_reports=True
        )
    )
    return optimizer


def print_convergence_reports(reports):
    for i_level, report in enumerate(reports):
        print("[LEVEL", i_level, "]")
        print(report)


def main():
    data_to_use = ds.PredefinedDatasetEnum.REAL3D_SNOOPY_SET05
    tsdf_generation_method = tsdf.GenerationMethod.EWA_TSDF_INCLUSIVE_CPP
    visualize_and_save_initial_and_final_fields = True
    out_path = "output/ho/single"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sampling.set_focus_coordinates(0, 0)
    generate_test_data = False

    live_field, canonical_field = \
        ds.datasets[data_to_use].generate_2d_sdf_fields(method=tsdf_generation_method, smoothing_coefficient=0.5)
    view_scaling_factor = 1024 // ds.datasets[data_to_use].field_size

    if visualize_and_save_initial_and_final_fields:
        viz.visualize_and_save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor)

    if generate_test_data:
        live_field = live_field[36:52, 21:37].copy()
        canonical_field = canonical_field[36:52, 21:37].copy()

    # optimizer = make_python_optimizer(out_path)
    optimizer = make_cpp_optimizer(out_path)
    warp_field = optimizer.optimize(canonical_field, live_field)

    print_convergence_reports(optimizer.get_per_level_convergence_reports())

    resampled_live = resampling.resample_field(live_field, warp_field)

    if visualize_and_save_initial_and_final_fields:
        viz.visualize_final_fields(canonical_field, resampled_live, view_scaling_factor)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
