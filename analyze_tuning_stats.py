#  ================================================================
#  Created by Gregory Kramida on 9/20/18.
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
import sys
import os.path

import yaml
import numpy as np
from matplotlib import pyplot as plt

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def print_at(index, name, data_energy_tensor, parameter_collection, iteration_count_tensor):
    min_data_energy_at = np.unravel_index(index, data_energy_tensor.shape)
    min_data_energy = data_energy_tensor[min_data_energy_at]
    data_term_weight = parameter_collection[0][min_data_energy_at[0]]
    smoothing_term_weight = parameter_collection[1][min_data_energy_at[1]]
    sobolev_kernel_size = parameter_collection[2][min_data_energy_at[2]]
    sobolev_kernel_strength = parameter_collection[3][min_data_energy_at[3]]
    iteration_count = iteration_count_tensor[min_data_energy_at]

    print("{:s}: {:f}, at run {:d}, with data term weight {:f},"
          " smoothing term weight {:f}, sobolev_kernel_size {:d}, sobolev_kernel_strength {:f}, iteration_count {:d}"
          .format(name, min_data_energy, index, data_term_weight, smoothing_term_weight,
                  sobolev_kernel_size, sobolev_kernel_strength, iteration_count))


def print_iteration_range_min(upper_bound, lower_bound, data_energy_tensor, iteration_count_tensor,
                              parameter_collection):
    cutoff_data_energies = data_energy_tensor.copy()
    cutoff_data_energies[np.where(iteration_count_tensor >= upper_bound)] = np.nan
    cutoff_data_energies[np.where(iteration_count_tensor < lower_bound)] = np.nan
    if np.isnan(cutoff_data_energies).sum() != cutoff_data_energies.size:
        item_index = np.nanargmin(cutoff_data_energies)
        print_at(item_index, "Minimal data energy, {:d} < iterations < {:d}".format(lower_bound, upper_bound),
                 data_energy_tensor, parameter_collection, iteration_count_tensor)


def main():
    data_term_weights = [0.2, 0.3, 0.6]
    smoothing_term_weights = [0.1, 0.2, 0.3]
    sobolev_kernel_sizes = [3, 7, 9]
    sobolev_kernel_strengths = [0.1, 0.15]
    parameter_collection = [data_term_weights, smoothing_term_weights, sobolev_kernel_sizes, sobolev_kernel_strengths]

    result_tensor_size = (
        len(data_term_weights), len(smoothing_term_weights), len(sobolev_kernel_sizes), len(sobolev_kernel_strengths)
    )

    data_energy_tensor = np.zeros(result_tensor_size, np.float32)
    data_energy_tensor.fill(np.nan)
    iteration_count_tensor = np.ndarray(result_tensor_size, np.int32)
    iteration_count_tensor.fill(np.nan)
    data_energies = []
    properly_read_runs = []

    run_count = data_energy_tensor.size

    current_run = 0
    read_runs_count = 0

    total_to_read_count = len(os.listdir("/media/algomorph/Data/Reconstruction/out_2D_SobolevFusionTuning"))
    print("Reading result files (seeing {:d} total runs with output)...".format(total_to_read_count))

    i_data_term_weight = 0
    for data_term_weight in data_term_weights:
        i_smoothing_term_weight = 0
        for smoothing_term_weight in smoothing_term_weights:
            i_sobolev_kernel_size = 0
            for sobolev_kernel_size in sobolev_kernel_sizes:
                i_sobolev_kernel_strength = 0
                for sobolev_kernel_strength in sobolev_kernel_strengths:
                    out_path = os.path.join("/media/algomorph/Data/Reconstruction/out_2D_SobolevFusionTuning",
                                            "run{:0>6d}".format(current_run))
                    results = None
                    input_parameters = None
                    if os.path.exists(out_path):
                        print_progress_bar(read_runs_count + 1, total_to_read_count,
                                           prefix="Progress: ", suffix="Complete", length=110)
                        try:
                            with open(os.path.join(out_path, "results.yaml"), 'r') as yaml_file:
                                try:
                                    results = yaml.load(yaml_file)
                                except yaml.YAMLError as exc:
                                    read_runs_count += 1
                            with open(os.path.join(out_path, "input_parameters.yaml"), 'r') as yaml_file:
                                try:
                                    input_parameters = yaml.load(yaml_file)
                                except yaml.YAMLError as exc:
                                    read_runs_count += 1
                        except FileNotFoundError as exc:
                            read_runs_count += 1
                    if results is not None and input_parameters is not None:
                        coordinates = (i_data_term_weight, i_sobolev_kernel_size,
                                       i_sobolev_kernel_size, i_sobolev_kernel_strength)
                        data_energy = results["final_data_energy"] / input_parameters["data_term_weight"]
                        data_energy_tensor[coordinates] = data_energy
                        data_energies.append(data_energy)
                        iteration_count_tensor[coordinates] = results["iterations"]
                        read_runs_count += 1
                    i_sobolev_kernel_strength += 1
                    current_run += 1
                i_sobolev_kernel_size += 1
            i_smoothing_term_weight += 1
        i_data_term_weight += 1

    print("Read {:d} results out of considered {:d} parameter combinations.".format(read_runs_count, run_count))
    item_index = np.nanargmin(data_energy_tensor)
    print_at(item_index, "Minimal data energy", data_energy_tensor, parameter_collection,
             iteration_count_tensor)

    for lower_bound in range(0, 35, 5):
        print_iteration_range_min(100, lower_bound, data_energy_tensor, iteration_count_tensor, parameter_collection)

    # plt.figure(figsize=(15, 10))
    # plt.plot(data_energies, "g")
    # plt.title("Data energy per run")
    # plt.savefig("")
    # plt.clf()
    # plt.close()

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
