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

# test targets
import hns_optimizer2d as hnso
import dataset as ds
import tsdf_field_generation as tsdf


class HNSOptimizerTest(TestCase):
    def test_construction_and_operation(self):
        data_to_use = ds.PredefinedDatasetEnum.REAL3D_SNOOPY_SET04
        depth_interpolation_method = tsdf.DepthInterpolationMethod.NONE

        live_field, canonical_field = \
            ds.datasets[data_to_use].generate_2d_sdf_fields(method=depth_interpolation_method)

        optimizer = hnso.HierarchicalNonrigidSLAMOptimizer2d(
            verbosity_parameters=hnso.HierarchicalNonrigidSLAMOptimizer2d.VerbosityParameters(
                print_max_warp_update=True
            ))
        optimizer.optimize(canonical_field, live_field)

        self.assertTrue(True)
