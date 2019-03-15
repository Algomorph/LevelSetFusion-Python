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
import os.path

# libraries
import numpy as np

# test data
from tests.test_data.hnso_fixtures import live_field, canonical_field, warp_field, final_live_field

# test targets
from nonrigid_opt import hierarchical_optimizer2d as hnso
from utils import field_resampling as resampling
from nonrigid_opt.sobolev_filter import generate_1d_sobolev_kernel
import experiment.dataset as dataset
import tsdf.common
import utils.path

# C++ extension
import level_set_fusion_optimization as cpp_extension


class HNSOptimizerTest(TestCase):
    def test_construction_and_operation(self):
        optimizer = hnso.HierarchicalOptimizer2d(
            rate=0.2,
            data_term_amplifier=1.0,
            maximum_warp_update_threshold=0.001,
            maximum_iteration_count=100,
            tikhonov_term_enabled=False,
            kernel=None,
            verbosity_parameters=hnso.HierarchicalOptimizer2d.VerbosityParameters(
                print_max_warp_update=False
            ))
        warp_field_out = optimizer.optimize(canonical_field, live_field)
        final_live_resampled = resampling.resample_field(live_field, warp_field_out)

        self.assertTrue(np.allclose(warp_field_out, warp_field))
        self.assertTrue(np.allclose(final_live_resampled, final_live_field))

        optimizer = cpp_extension.HierarchicalOptimizer(
            tikhonov_term_enabled=False,
            gradient_kernel_enabled=False,
            maximum_chunk_size=8,
            rate=0.2,
            maximum_iteration_count=100,
            maximum_warp_update_threshold=0.001,
            data_term_amplifier=1.0
        )

        warp_field_out = optimizer.optimize(canonical_field, live_field)
        final_live_resampled = resampling.resample_field(live_field, warp_field_out)
        self.assertTrue(np.allclose(warp_field_out, warp_field, atol=10e-6))
        self.assertTrue(np.allclose(final_live_resampled, final_live_field, atol=10e-6))


    # TODO finish
    # def test_operation1(self):
    #     dataset_to_use = dataset.PredefinedDatasetEnum.REAL3D_SNOOPY_SET05
    #     generation_method = tsdf.common.GenerationMethod.EWA_TSDF_INCLUSIVE_CPP
    #
    #     canonical_field, live_field = dataset.datasets[dataset_to_use].generate_2d_sdf_fields(generation_method)
    #
    #     verbosity_parameters = cpp_extension.HierarchicalOptimizer.VerbosityParameters(
    #         print_max_warp_update=False,
    #         print_iteration_mean_tsdf_difference=False,
    #         print_iteration_std_tsdf_difference=False,
    #         print_iteration_data_energy=False,
    #         print_iteration_tikhonov_energy=False)
    #
    #     logging_parameters = cpp_extension.HierarchicalOptimizer.LoggingParameters(
    #         collect_per_level_convergence_reports=True
    #     )
    #
    #     optimizer = cpp_extension.HierarchicalOptimizer(
    #         tikhonov_term_enabled=False,
    #         gradient_kernel_enabled=False,
    #
    #         maximum_chunk_size=8,
    #         rate=0.2,
    #         maximum_iteration_count=100,
    #         maximum_warp_update_threshold=0.001,
    #
    #         data_term_amplifier=1.0,
    #         tikhonov_strength=0.2,
    #         kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),
    #         verbosity_parameters=verbosity_parameters,
    #         logging_parameters=logging_parameters
    #     )

