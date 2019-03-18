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
from experiment import dataset as ds
from tsdf import generation as tsdf
from nonrigid_opt import field_warping as resampling
import utils.sampling as sampling
import experiment.build_hierarchical_optimizer_helper as build_opt
# has to be compiled and included in PYTHONPATH first
import level_set_fusion_optimization as ho_cpp

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def print_convergence_reports(reports):
    for i_level, report in enumerate(reports):
        print("[LEVEL", i_level, "]")
        print(report)


def main():
    data_to_use = ds.PredefinedDatasetEnum.REAL3D_SNOOPY_SET00
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

    shared_parameters = build_opt.HierarchicalOptimizer2dSharedParameters()
    shared_parameters.maximum_warp_update_threshold = 0.01
    verbosity_parmeters_py = build_opt.make_common_hierarchical_optimizer2d_py_verbosity_parameters()
    verbosity_parameters_cpp = ho_cpp.HierarchicalOptimizer2d.VerbosityParameters(
        print_max_warp_update=True,
        print_iteration_mean_tsdf_difference=True,
        print_iteration_std_tsdf_difference=True,
        print_iteration_data_energy=True,
        print_iteration_tikhonov_energy=True,
    )
    visualization_parameters_py = build_opt.make_common_hierarchical_optimizer2d_visualization_parameters()
    visualization_parameters_py.out_path = out_path
    logging_parameters_cpp = ho_cpp.HierarchicalOptimizer2d.LoggingParameters(
        collect_per_level_convergence_reports=True)
    optimizer = build_opt.make_hierarchical_optimizer2d(implementation_language=build_opt.ImplementationLanguage.CPP,
                                                        shared_parameters=shared_parameters,
                                                        verbosity_parameters_cpp=verbosity_parameters_cpp,
                                                        logging_parameters_cpp=logging_parameters_cpp,
                                                        verbosity_parameters_py=verbosity_parmeters_py,
                                                        visualization_parameters_py=visualization_parameters_py)
    warp_field = optimizer.optimize(canonical_field, live_field)

    print_convergence_reports(optimizer.get_per_level_convergence_reports())

    resampled_live = resampling.warp_field(live_field, warp_field)

    if visualize_and_save_initial_and_final_fields:
        viz.visualize_final_fields(canonical_field, resampled_live, view_scaling_factor)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
