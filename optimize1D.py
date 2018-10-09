#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 11/14/17.
#  Copyright (c) 2017 Gregory Kramida
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

import numpy as np
import math
import random
import sys

from matplotlib import pyplot as plt

from enum import Enum

IGNORE_OPENCV = False

try:
    from cv2 import VideoWriter
    import cv2

except ImportError:
    IGNORE_OPENCV = True


EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1

PRINT_RANGE_START = 25
PRINT_RANGE_END = 60

BOLD_YELLOW = "\033[33;1;m"
BOLD_GREEN = "\033[32;1;m"
BOLD_BLUE = "\033[34;1;m"
RESET = "\033[0m"

SDF_VOXEL_RATIO = 10

# Haha, global vars are bad juju, but here -- who cares?
global_iteration = 0


class Schema(Enum):
    WARP_WARP = 1
    UPDATE_UPDATE = 2
    WARP_UPDATE = 3


def dx(scalar_field):
    """
    Computes first derivative of a 1D scalar field
    :param scalar_field:
    :return:
    """

    first_derivative = np.zeros((scalar_field.size - 1))
    for i_scalar in range(scalar_field.size - 1):
        i_next_scalar = i_scalar + 1
        first_derivative[i_scalar] = scalar_field[i_next_scalar] - scalar_field[i_scalar]
    return first_derivative


def dxdxx(scalar_field):
    """
    compute second derivative of a 1D scalar field
    :param scalar_field:
    :return:
    :rtype:(numpy.ndarray,numpy.ndarray)
    """
    first_derivative = dx(scalar_field)
    second_derivative = dx(first_derivative)
    return first_derivative, second_derivative


def old_test():
    scalar_field_size = 10
    initial_scalar_field = np.random.rand(scalar_field_size)
    print(initial_scalar_field)
    scalar_field = initial_scalar_field.copy()
    learning_rate = 0.1

    max_update = np.inf
    update_threshold = 0.0001
    iteration_count = 0

    while np.abs(max_update) > update_threshold:
        first_derivative, second_derivative = dxdxx(scalar_field)
        update = np.zeros_like(second_derivative)
        for i_update in range(update.size):
            update[i_update] = -2.0 * (first_derivative[i_update + 1] + second_derivative[i_update])
            # update[i_update] = -2.0 * second_derivative[i_update]
        scalar_field[1:-1] -= learning_rate * update
        max_update = np.max(update)
        print(max_update)
        iteration_count += 1

    print(scalar_field)
    print("Iteration count: ", iteration_count)


def generate_sample_1d_sdf_scalar_field(surface_count, size, narrow_band_width_voxels=20,
                                        count_of_negative_voxels_behind_surface=20):
    surface_locations = []
    random.seed()
    half_width = narrow_band_width_voxels // 2
    maximum_tries = 3000
    tries_so_far = 0

    while len(surface_locations) < surface_count:
        location = random.uniform(half_width, size - half_width)
        too_close = False
        for another_location in surface_locations:
            if abs(location - another_location) < (half_width + 1):
                too_close = True
                break
        if not too_close:
            surface_locations.append(location)
        tries_so_far += 1
        if tries_so_far > maximum_tries:
            raise ValueError("Too many tries for generating spikes: ether size is not enough to accommodate the given "
                             "surface count or number of surfaces is too large for the test to be run in adequate "
                             "time")
    surface_locations = sorted(surface_locations)
    field = np.ones(size, dtype=np.float32)

    for location in surface_locations:
        start_point = int(location - half_width)
        end_point = int(location + half_width + count_of_negative_voxels_behind_surface + 1)
        for i_voxel in range(start_point, end_point):
            distance = min(max((location - i_voxel) / half_width, -1.0), 1.0)
            field[i_voxel] = distance
    return field, surface_locations


def generate_1d_sdf_scalar_field_1_surface(location, size, narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    field = np.ones(size, dtype=np.float32)
    half_width = narrow_band_width_voxels // 2
    start_point = int(location - half_width)
    end_point = int(location + min(half_width, back_cutoff_voxels) + 1)
    if location - narrow_band_width_voxels < 0:
        raise ValueError("Location of surface too close to 0 for a full narrow band representation")
    for i_voxel in range(start_point, end_point):
        distance = min(max((location - i_voxel) / half_width, -1.0), 1.0)
        field[i_voxel] = distance

    # fill the rest with -1.0
    if end_point < size and end_point < back_cutoff_voxels:
        field[end_point:] = -1.0
    return field


def shift2(arr, num, fill_val=0):
    arr = np.roll(arr, num)
    if num < 0:
        np.put(arr, range(len(arr) + num, len(arr)), fill_val)
    elif num > 0:
        np.put(arr, range(num), fill_val)
    return arr


def shift_1d_scalar_field(scalar_field, offset=5):
    new_field = scalar_field.copy()
    shift2(new_field, scalar_field, fill_val=1.0)


def data_term_at_location(warped_live_field, canonical_field, location):
    normalize_gradient = False

    live_sdf = warped_live_field[location] * SDF_VOXEL_RATIO
    canonical_sdf = canonical_field[location] * SDF_VOXEL_RATIO

    diff = live_sdf - canonical_sdf

    if location > 0:
        live_previous = warped_live_field[location - 1] * SDF_VOXEL_RATIO
    else:
        live_previous = 1.0
    if location < len(warped_live_field) - 1:
        live_next = warped_live_field[location + 1] * SDF_VOXEL_RATIO
    else:
        live_next = 1.0
    live_local_gradient = 0.5 * (live_next - live_previous)
    if normalize_gradient:
        live_local_gradient = np.sign(live_local_gradient)
    # unscaled_warp_gradient_contribution = diff * live_local_gradient
    # if abs(live_local_gradient) > 10e-4 and abs(diff) > 10e-3:
    if live_local_gradient != 0.0:
        # if location == 56:
        #     print("diff:", diff)
        #     print("llg:", live_local_gradient)
        unscaled_warp_gradient_contribution = diff / live_local_gradient
    else:
        unscaled_warp_gradient_contribution = 0
    local_energy_contribution = 0.5 * pow(diff, 2)
    return unscaled_warp_gradient_contribution, local_energy_contribution


def smoothing_term_at_location(warp_field, location):
    # 1D discrete laplacian using finite-differences
    warp = warp_field[location]
    warp_before = warp if location == 0 else warp_field[location - 1]
    warp_after = warp if location == len(warp_field) - 1 else warp_field[location + 1]
    unscaled_warp_gradient_contribution = -1 * (warp_before - 2 * warp + warp_after)

    warp_gradient = 0.5 * (warp_after - warp_before)
    local_energy_contribution = 0.5 * pow(warp_gradient, 2)
    return unscaled_warp_gradient_contribution, local_energy_contribution


def interpolate_warped_live(warped_live_field, warp_field):
    field_size = warp_field.shape[0]
    new_warped_live_field = np.ones_like(warped_live_field)
    for location in range(0, field_size):
        warped_location = location + warp_field[location]
        if warped_location < 0 or warped_location > field_size - 1:
            continue  # out of bounds
        first_sample_at = int(math.floor(warped_location))
        second_sample_at = first_sample_at + 1
        ratio = warped_location - first_sample_at
        if second_sample_at > field_size - 1:
            second_sample_at = field_size - 1
        sample_point1 = warped_live_field[first_sample_at]
        sample_point2 = warped_live_field[second_sample_at]
        interpolated_value = sample_point1 * (1.0 - ratio) + sample_point2 * ratio
        if 1.0 - abs(interpolated_value) < 10e-4:
            interpolated_value = np.sign(interpolated_value)
            warp_field[location] = 0.0
        new_warped_live_field[location] = interpolated_value
    warped_live_field[:] = new_warped_live_field[:]


def print_1d_range(field, from_index_inclusive, to_index_exclusive, color_code=BOLD_GREEN):
    for i_value in range(from_index_inclusive, to_index_exclusive):
        sys.stdout.write("[")
        sys.stdout.write(str(i_value))
        sys.stdout.write(":")
        sys.stdout.write(color_code)
        sys.stdout.write('{:+05f}'.format(field[i_value]))
        sys.stdout.write(RESET)
        sys.stdout.write("]")
        sys.stdout.flush()
        # print("[", i_value, ":", field[i_value], "]", end='')
    print()


class Optimizer:
    def __init__(self):
        self.iteration_number = 0
        self.total_data_energy = 0.
        self.total_smoothing_energy = 0.
        self.gradient_descent_rate = 0.1
        self.smoothing_term_weight = 0.2
        self.warp_length_termination_threshold = 0.025
        self.max_iterations = 200
        self.min_iterations = 0
        self.record_video = True
        self.schema = Schema.UPDATE_UPDATE

    def optimization_iteration(self, warped_live_field, canonical_field, warp_field, update_field):
        self.total_data_energy = 0.
        self.total_smoothing_energy = 0.
        band_union_only = True

        if warped_live_field.shape != canonical_field.shape or warped_live_field.shape != warp_field.shape or \
                len(warp_field.shape) != 1:
            raise ValueError(
                "warp field, warped live field, and canonical field all need to be 1D arrays of the same size.")
        field_size = warp_field.shape[0]

        for location in range(0, field_size):
            if band_union_only:
                sdf_live = warped_live_field[location]
                sdf_canonical = canonical_field[location]
                if (sdf_live == 1.0 or sdf_live == -1.0) and sdf_canonical == 1.0:
                    update_field[location] = 0.0
                    continue
            update = 0.0

            data_contribution, local_data_energy = data_term_at_location(warped_live_field, canonical_field, location)
            self.total_data_energy += local_data_energy
            update += data_contribution

            smoothing_contribution, local_smoothing_energy = smoothing_term_at_location(warp_field, location)
            self.total_smoothing_energy += self.smoothing_term_weight * local_smoothing_energy
            update += self.smoothing_term_weight * smoothing_contribution

            update_field[location] = update

        max_warp = 0.0
        max_warp_location = -1
        max_update = 0.0
        max_update_location = -1
        for location in range(0, field_size):
            if self.schema == Schema.WARP_WARP or self.schema == Schema.WARP_UPDATE:
                warp_field[location] -= update_field[location] * self.gradient_descent_rate
            elif self.schema == Schema.UPDATE_UPDATE:
                warp_field[location] = -update_field[location] * self.gradient_descent_rate
            if abs(warp_field[location]) > max_warp:
                max_warp = abs(warp_field[location])
                max_warp_location = location
            if abs(update_field[location]) > max_update:
                max_update = abs(update_field[location])
                max_update_location = location

        if self.schema == Schema.WARP_UPDATE:
            interpolate_warped_live(warped_live_field, update_field)
        elif self.schema == Schema.UPDATE_UPDATE or self.schema == Schema.WARP_WARP:
            interpolate_warped_live(warped_live_field, warp_field)

        return max_warp, max_warp_location, max_update, max_update_location

    def optimize(self, live_field, canonical_field, warp_field):
        max_warp = np.inf
        max_update = np.inf
        self.iteration_number = 0
        update_field = np.zeros_like(warp_field)
        x_axis = np.arange(0, len(live_field))
        if self.record_video and not IGNORE_OPENCV:
            live_video_writer = cv2.VideoWriter("live_field_evolution_1D.mkv",
                                                cv2.VideoWriter_fourcc('X', '2', '6', '4'),
                                                10, (640, 480), isColor=True)
            warp_video_writer = cv2.VideoWriter("warp_magnitudes_1D.mkv",
                                                cv2.VideoWriter_fourcc('X', '2', '6', '4'),
                                                10, (640, 480), isColor=True)  # type: VideoWriter

        while self.iteration_number < self.min_iterations or (
                max_warp > self.warp_length_termination_threshold and self.iteration_number < self.max_iterations):
            max_warp, max_warp_location, max_update, max_update_location = \
                self.optimization_iteration(live_field, canonical_field, warp_field, update_field)

            if self.record_video and not IGNORE_OPENCV:
                plt.clf()
                fig = plt.figure()
                plt.plot(x_axis, canonical_field, color='black')
                plt.plot(x_axis, live_field, color='green')
                plt.title("warped live " + str(self.iteration_number))

                fig.canvas.draw()
                plt.close()
                plot_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                live_video_writer.write(plot_image)

                plt.clf()
                fig = plt.figure()
                plt.axis([0, 100, -0.6, 0.6])
                plt.plot(x_axis, warp_field, color='red')
                plt.title("warp field " + str(self.iteration_number))

                fig.canvas.draw()
                plt.close()
                plot_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                warp_video_writer.write(plot_image)

            print("[Iteration ", self.iteration_number, "] data energy:", self.total_data_energy, "; smoothing energy:",
                  self.total_smoothing_energy, "; total energy:", self.total_data_energy + self.total_smoothing_energy,
                  "; max warp:", max_warp, "at", max_warp_location, "; max update:",
                  max_update, "at", max_update_location)
            print_1d_range(canonical_field, PRINT_RANGE_START, PRINT_RANGE_END, BOLD_YELLOW)
            print_1d_range(live_field, PRINT_RANGE_START, PRINT_RANGE_END, BOLD_GREEN)
            print_1d_range(warp_field, PRINT_RANGE_START, PRINT_RANGE_END, BOLD_BLUE)
            print()
            self.iteration_number += 1
            global_iteration = self.iteration_number
        if self.record_video and not IGNORE_OPENCV:
            live_video_writer.release()
            warp_video_writer.release()


def main():
    field_size = 100

    mimic_delta = False

    canonical_at = 47
    live_at = 42

    if mimic_delta:
        canonical_field = generate_1d_sdf_scalar_field_1_surface(canonical_at, field_size, back_cutoff_voxels=3)
    else:
        canonical_field = generate_1d_sdf_scalar_field_1_surface(canonical_at, field_size)
    live_field = generate_1d_sdf_scalar_field_1_surface(live_at, field_size)
    warp_field = np.zeros(field_size, dtype=np.float32)

    print("Canonical:")
    print_1d_range(canonical_field, PRINT_RANGE_START, PRINT_RANGE_END)
    print("Live:")
    print_1d_range(live_field, PRINT_RANGE_START, PRINT_RANGE_END)
    print()

    optimizer = Optimizer()
    optimizer.optimize(live_field, canonical_field, warp_field)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
