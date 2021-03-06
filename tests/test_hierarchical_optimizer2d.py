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
from unittest import TestCase

# libraries
import numpy as np

# test data
import tests.test_data.hierarchical_optimizer_test_data as test_data

# test targets
from nonrigid_opt.hierarchical import hierarchical_optimization_visualizer as hov_py, hierarchical_optimizer2d as ho_py
from nonrigid_opt import field_warping as resampling
import experiment.dataset as dataset
import tsdf.common
import experiment.hierarchical_optimizer.build_helper as build_opt
import nonrigid_opt.slavcheva.sobolev_filter as sob

# C++ extension
import level_set_fusion_optimization as ho_cpp


class HierarchicalOptimizerTest(TestCase):
    def test_construction_and_operation01(self):
        optimizer = ho_py.HierarchicalOptimizer2d(
            rate=0.2,
            data_term_amplifier=1.0,
            maximum_warp_update_threshold=0.001,
            maximum_iteration_count=100,
            tikhonov_term_enabled=False,
            kernel=None,
            verbosity_parameters=ho_py.HierarchicalOptimizer2d.VerbosityParameters(
                print_max_warp_update=False
            ))
        warp_field_out = optimizer.optimize(test_data.canonical_field, test_data.live_field)
        final_warped_live = resampling.warp_field(test_data.live_field, warp_field_out)

        self.assertTrue(np.allclose(warp_field_out, test_data.warp_field))
        self.assertTrue(np.allclose(final_warped_live, test_data.final_live_field))

        optimizer = ho_cpp.HierarchicalOptimizer2d(
            tikhonov_term_enabled=False,
            gradient_kernel_enabled=False,
            maximum_chunk_size=8,
            rate=0.2,
            maximum_iteration_count=100,
            maximum_warp_update_threshold=0.001,
            data_term_amplifier=1.0
        )

        warp_field_out = optimizer.optimize(test_data.canonical_field, test_data.live_field)
        final_warped_live = resampling.warp_field(test_data.live_field, warp_field_out)
        self.assertTrue(np.allclose(warp_field_out, test_data.warp_field, atol=10e-6))
        self.assertTrue(np.allclose(final_warped_live, test_data.final_live_field, atol=10e-6))

    def test_cpp_iteration_data(self):
        optimizer = ho_cpp.HierarchicalOptimizer2d(
            tikhonov_term_enabled=False,
            gradient_kernel_enabled=False,

            maximum_chunk_size=8,
            rate=0.2,
            maximum_iteration_count=100,
            maximum_warp_update_threshold=0.001,

            data_term_amplifier=1.0,
            tikhonov_strength=0.0,

            kernel=sob.generate_1d_sobolev_kernel(size=7, strength=0.1),

            resampling_strategy=ho_cpp.HierarchicalOptimizer2d.ResamplingStrategy.NEAREST_AND_AVERAGE,

            verbosity_parameters=ho_cpp.HierarchicalOptimizer2d.VerbosityParameters(),
            logging_parameters=ho_cpp.HierarchicalOptimizer2d.LoggingParameters(
                collect_per_level_convergence_reports=True,
                collect_per_level_iteration_data=True
            )
        )
        warp_field_out = optimizer.optimize(test_data.canonical_field, test_data.live_field)
        final_warped_live = resampling.warp_field(test_data.live_field, warp_field_out)
        data = optimizer.get_per_level_iteration_data()
        vec = data[3].get_warp_fields()

        self.assertTrue(np.allclose(vec[50], test_data.iteration50_warp_field, atol=1e-6))

        self.assertTrue(np.allclose(warp_field_out, test_data.warp_field, atol=10e-6))
        self.assertTrue(np.allclose(final_warped_live, test_data.final_live_field, atol=10e-6))

    def test_construction_and_operations02(self):
        dataset_to_use = dataset.PredefinedDatasetEnum.REAL3D_SNOOPY_SET00
        generation_method = ho_cpp.tsdf.FilteringMethod.EWA_VOXEL_SPACE_INCLUSIVE

        camera_intrinsic_matrix = np.array([[700., 0., 320.],
                                            [0., 700., 240.],
                                            [0., 0., 1.]], dtype=np.float32)

        canonical_field, live_field = dataset.datasets[dataset_to_use].generate_2d_sdf_fields(generation_method,
                                                                                              use_cpp=True)

        shared_parameters = build_opt.HierarchicalOptimizer2dSharedParameters()
        shared_parameters.maximum_warp_update_threshold = 0.01
        shared_parameters.maximum_iteration_count = 2

        # Python-specific
        verbosity_parameters_py = ho_py.HierarchicalOptimizer2d.VerbosityParameters()
        visualization_parameters_py = hov_py.HierarchicalOptimizer2dVisualizer.Parameters()
        visualization_parameters_py.out_path = "out"

        # C++-specific
        verbosity_parameters_cpp = ho_cpp.HierarchicalOptimizer2d.VerbosityParameters()
        logging_parameters_cpp = ho_cpp.HierarchicalOptimizer2d.LoggingParameters(
            collect_per_level_convergence_reports=True,
            collect_per_level_iteration_data=False
        )
        resampling_strategy = ho_cpp.HierarchicalOptimizer2d.ResamplingStrategy.NEAREST_AND_AVERAGE

        optimizer_cpp = build_opt.make_hierarchical_optimizer2d(
            implementation_language=build_opt.ImplementationLanguage.CPP,
            shared_parameters=shared_parameters,
            verbosity_parameters_cpp=verbosity_parameters_cpp,
            logging_parameters_cpp=logging_parameters_cpp,
            verbosity_parameters_py=verbosity_parameters_py,
            visualization_parameters_py=visualization_parameters_py,
            resampling_strategy_cpp=resampling_strategy
        )

        warp_field_cpp = optimizer_cpp.optimize(canonical_field, live_field)
        warped_live_cpp = resampling.warp_field(live_field, warp_field_cpp)

        optimizer_py = build_opt.make_hierarchical_optimizer2d(
            implementation_language=build_opt.ImplementationLanguage.PYTHON,
            shared_parameters=shared_parameters,
            verbosity_parameters_cpp=verbosity_parameters_cpp,
            logging_parameters_cpp=logging_parameters_cpp,
            verbosity_parameters_py=verbosity_parameters_py,
            visualization_parameters_py=visualization_parameters_py)

        warp_field_py = optimizer_py.optimize(canonical_field, live_field)
        warped_live_py = resampling.warp_field(live_field, warp_field_py)

        self.assertTrue(np.allclose(warp_field_cpp, warp_field_py, atol=10e-6))
        self.assertTrue(np.allclose(warped_live_cpp, warped_live_py, atol=10e-6))
