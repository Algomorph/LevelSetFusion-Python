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
# local
import hns_optimizer2d as hnso
import hns_visualizer as hnsov
import dataset as ds
import tsdf_field_generation as tsdf

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1

def main():
    data_to_use = ds.PredefinedDatasetEnum.REAL3D_SNOOPY_SET04
    depth_interpolation_method = tsdf.DepthInterpolationMethod.NONE
    out_path = "output/hnso"

    live_field, canonical_field = \
        ds.datasets[data_to_use].generate_2d_sdf_fields(method=depth_interpolation_method)

    optimizer = hnso.HierarchicalNonrigidSLAMOptimizer2d(
        rate=0.2,
        data_term_amplifier=1.0,
        verbosity_parameters=hnso.HierarchicalNonrigidSLAMOptimizer2d.VerbosityParameters(
            print_iteration_data_energy=True,
            print_max_warp_update=True,
        ),
        visualization_parameters=hnsov.HNSOVisualizer.Parameters(
            out_path=out_path,
            save_live_progression=True,
            save_initial_fields=True,
            save_final_fields=True,
            save_warp_field_progression=True
        )


    )
    optimizer.optimize(canonical_field, live_field)
    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
