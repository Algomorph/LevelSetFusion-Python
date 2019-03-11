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
import os.path

# libraries
import numpy as np
import progressbar

# local
import experiment.path_utility as pu
import experiment.experiment_shared_routines as esr
from tsdf import generation as tsdf
from utils import field_resampling as resampling
import utils.sampling as sampling
from nonrigid_opt.sobolev_filter import generate_1d_sobolev_kernel
import utils.printing as prt

# has to be compiled and installed first (cpp folder)
import level_set_fusion_optimization as cpp_extension

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def main():
    # program argument candidates
    generation_method = tsdf.GenerationMethod.EWA_TSDF_INCLUSIVE_CPP

    out_path = "output/ho"

    frame_pair_datasets = esr.prepare_dataset_for_2d_frame_pair_processing(
        calibration_path=os.path.join(pu.get_reconstruction_directory(),
                                      "real_data/snoopy/snoopy_calib.txt"),
        frame_directory=os.path.join(pu.get_reconstruction_directory(),
                                     "real_data/snoopy/frames/"),
        output_directory=out_path,
        y_range=(214, 400),
        replace_empty_rows=True,
        use_masks=True,
        input_case_file=None,
        offset=np.array([-64, -64, 128]),
        field_size=128,
    )

    verbosity_parameters = cpp_extension.HierarchicalOptimizer.VerbosityParameters(
        print_max_warp_update=False,
        print_iteration_mean_tsdf_difference=False,
        print_iteration_std_tsdf_difference=False,
        print_iteration_data_energy=False,
        print_iteration_tikhonov_energy=False)

    optimizer = cpp_extension.HierarchicalOptimizer(
        tikhonov_term_enabled=False,
        gradient_kernel_enabled=False,

        maximum_chunk_size=8,
        rate=0.2,
        maximum_iteration_count=100,
        maximum_warp_update_threshold=0.001,

        data_term_amplifier=1.0,
        tikhonov_strength=0.2,
        kernel=generate_1d_sobolev_kernel(size=7, strength=0.1),
        verbosity_parameters=verbosity_parameters
    )

    pair_count = len(frame_pair_datasets)

    for dataset in progressbar.progressbar(frame_pair_datasets):
        canonical_field, live_field = dataset.generate_2d_sdf_fields(generation_method)
        warp_field_out = optimizer.optimize(canonical_field, live_field)
        final_live_resampled = resampling.resample_field(live_field, warp_field_out)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
