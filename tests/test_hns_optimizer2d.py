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

# test targets
from nonrigid_opt import hns_optimizer2d as hnso
from utils import field_resampling as resampling
from tests.hnso_fixtures import live_field, canonical_field, warp_field, final_live_field
# C++ extension
import level_set_fusion_optimization as cpp_extension


class HNSOptimizerTest(TestCase):
    def test_construction_and_operation(self):
        optimizer = hnso.HierarchicalNonrigidSLAMOptimizer2d(
            rate=0.2,
            data_term_amplifier=1.0,
            maximum_warp_update_threshold=0.001,
            maximum_iteration_count=100,
            tikhonov_term_enabled=False,
            kernel=None,
            verbosity_parameters=hnso.HierarchicalNonrigidSLAMOptimizer2d.VerbosityParameters(
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
