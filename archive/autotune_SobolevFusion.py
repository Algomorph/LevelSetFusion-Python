#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 9/19/18.
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
import os
import os.path
import sys

# common libs
import yaml
import json
import numpy as np

# local
from field_generator import generate_initial_fields
from slavcheva_optimizer2d import SlavchevaOptimizer2d, AdaptiveLearningRateMethod
from sobolev_filter import generate_1d_sobolev_kernel
from utils.printing import *

IGNORE_OPENCV = False

try:
    import cv2
except ImportError:
    IGNORE_OPENCV = True

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def main():
    visualize_and_save_initial_and_final_fields = False
    field_size = 128
    default_value = 0
    view_scaling_factor = 8

    live_field, canonical_field, warp_field = generate_initial_fields(field_size=field_size,
                                                                      live_smoothing_kernel_size=0,
                                                                      canonical_smoothing_kernel_size=0,
                                                                      default_value=default_value)

    start_from_run = 0

    data_term_weights = [0.2, 0.3, 0.6]
    smoothing_term_weights = [0.1, 0.2, 0.3]
    sobolev_kernel_sizes = [3, 7, 9]
    sobolev_kernel_strengths = [0.1, 0.15]

    total_number_of_runs = len(data_term_weights) * len(smoothing_term_weights) * \
                           len(sobolev_kernel_sizes) * len(sobolev_kernel_strengths)

    end_before_run = total_number_of_runs
    current_run = 0

    max_iterations = 100
    maximum_warp_length_lower_threshold = 0.1

    for data_term_weight in data_term_weights:
        for smoothing_term_weight in smoothing_term_weights:
            for sobolev_kernel_size in sobolev_kernel_sizes:
                for sobolev_kernel_strength in sobolev_kernel_strengths:
                    if current_run < start_from_run:
                        current_run += 1
                        continue

                    if current_run >= end_before_run:
                        current_run += 1
                        continue


                    print("{:s}STARTING RUN {:0>6d}{:s}".format(BOLD_LIGHT_CYAN, current_run, RESET))

                    input_parameters = {
                        "data_term_weight": float(data_term_weight),
                        "smoothing_term_weight": float(smoothing_term_weight),
                        "sobolev_kernel_size": int(sobolev_kernel_size),
                        "sobolev_kernel_strength": float(sobolev_kernel_strength),
                        "max_iterations": max_iterations,
                        "maximum_warp_length_lower_threshold": maximum_warp_length_lower_threshold
                    }
                    print("Input Parameters:")
                    print(json.dumps(input_parameters, sort_keys=True, indent=4))
                    out_path = os.path.join("/media/algomorph/Data/Reconstruction/out_2D_SobolevFusionTuning",
                                            "run{:0>6d}".format(current_run))
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    with open(os.path.join(out_path, "input_parameters.yaml"), 'w') as yaml_file:
                        yaml.dump(input_parameters, yaml_file, default_flow_style=False)

                    live_field_copy = live_field.copy()
                    canonical_field_copy = canonical_field.copy()
                    warp_field_copy = warp_field.copy()

                    optimizer = SlavchevaOptimizer2d(
                        out_path=out_path,
                        field_size=field_size,

                        data_term_weight=data_term_weight,
                        smoothing_term_weight=smoothing_term_weight,
                        level_set_term_weight=0.5,
                        sobolev_kernel=
                        generate_1d_sobolev_kernel(size=sobolev_kernel_size, strength=sobolev_kernel_strength),
                        level_set_term_enabled=False,

                        maximum_warp_length_lower_threshold=maximum_warp_length_lower_threshold,
                        max_iterations=max_iterations,

                        adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                        default_value=default_value,

                        enable_component_fields=True,
                        view_scaling_factor=view_scaling_factor)

                    optimizer.optimize(live_field_copy, canonical_field_copy, warp_field_copy)
                    optimizer.plot_logged_sdf_and_warp_magnitudes()
                    optimizer.plot_logged_energies_and_max_warps()

                    sdf_diff = float(np.sum((live_field - canonical_field)**2))

                    output_results = {
                        "sdf_diff": sdf_diff,
                        "iterations": len(optimizer.log.max_warps),
                        "final_max_warp_length": float(optimizer.log.max_warps[-1]),
                        "initial_data_energy": float(optimizer.log.data_energies[0]),
                        "final_data_energy": float(optimizer.log.data_energies[-1]),
                        "initial_energy": float(optimizer.log.data_energies[0] + optimizer.log.smoothing_energies[0]),
                        "final_energy": float(optimizer.log.data_energies[-1] + optimizer.log.smoothing_energies[-1]),
                        "initial_smoothing_energy": float(optimizer.log.smoothing_energies[0]),
                        "final_smoothing_energy": float(optimizer.log.smoothing_energies[-1])
                    }
                    print("Tuning Results:")
                    print(json.dumps(output_results, sort_keys=True, indent=4))
                    with open(os.path.join(out_path, "results.yaml"), 'w') as yaml_file:
                        yaml.dump(output_results, yaml_file, default_flow_style=False)

                    touch_path = os.path.join(out_path,
                                              "ran_for_{:4>d}_iterations".format(len(optimizer.log.max_warps)))
                    with open(touch_path, 'a'):
                        os.utime(touch_path)

                    touch_path = os.path.join(out_path,
                                              "sdf_diff_{:3.2f}".format(sdf_diff))
                    with open(touch_path, 'a'):
                        os.utime(touch_path)

                    print("{:s}FINISHED RUN {:0>6d}{:s}".format(BOLD_LIGHT_CYAN, current_run, RESET))
                    current_run += 1

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
