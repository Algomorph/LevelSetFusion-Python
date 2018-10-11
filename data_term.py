#  ================================================================
#  Created by Gregory Kramida on 9/17/18.
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
import numpy as np
from sampling import sample_at, sample_at_replacement, focus_coordinates_match, sample_flag_at
from printing import *
from scipy.signal import convolve2d
from enum import Enum
import level_set_fusion_optimization as lsfo


class FiniteDifferenceMethod(Enum):
    CENTRAL = 0
    FORWARD = 1
    BACK = 2


class DataTermMethod(Enum):
    BASIC = 0
    BASIC_CPP = 1
    THRESHOLDED_FDM = 2  # threshold determines the finite-difference method


gaussian_kernel3x3_sigma1 = np.array([[0.077847, 0.123317, 0.077847],
                                      [0.123317, 0.195346, 0.123317],
                                      [0.077847, 0.123317, 0.077847]], dtype=np.float32)
gaussian_kernel3x3_sigmaPoint8 = np.array([[0.0625, 0.125, 0.0625],
                                           [0.125, 0.25, 0.125],
                                           [0.0625, 0.125, 0.0625]], dtype=np.float32)
gaussian_kernel3x3 = gaussian_kernel3x3_sigmaPoint8


def sampling_convolve2d(field, kernel, x, y):
    ans = 0.0
    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            ans += sample_at(field, y - 1 + ky, x - 1 + kx) * kernel[ky, kx]
    return ans


def print_gradient_data(live_y_minus_one, live_x_minus_one, live_y_plus_one, live_x_plus_one, central_value):
    print()
    print("[Grad data         ]", BOLD_BLUE, sep='')
    print("                     ", "      {:+01.3f}".format(live_y_minus_one), sep='')
    print("                     ",
          "{:+01.3f}{:+01.3f}{:+01.3f}".format(live_x_minus_one, central_value, live_x_plus_one), sep='')
    print("                     ", "      {:+01.3f}".format(live_y_plus_one), RESET, sep='')


def print_gradient_data_3x3_receptive_field(field, x, y, live_y_minus_one, live_x_minus_one, live_y_plus_one,
                                            live_x_plus_one, central_value):
    print()
    line0 = ""
    line1 = "                           {:s}{:+01.3f}{:s}".format(BOLD_BLUE, live_y_minus_one, RESET)
    line2 = " ==================> {:s}{:+01.3f}{:+01.3f}{:+01.3f}{:s}".format(BOLD_BLUE, live_x_minus_one,
                                                                              central_value, live_x_plus_one, RESET)
    line3 = "                           {:s}{:+01.3f}{:s}".format(BOLD_BLUE, live_y_plus_one, RESET)
    line4 = ""
    end_lines = [line0, line1, line2, line3, line4]

    receptive_field_start_x = x - 2
    receptive_field_start_y = y - 2
    receptive_field_end_x = x + 3
    receptive_field_end_y = y + 3

    draw_as_blank = {(receptive_field_start_x, receptive_field_start_y),
                     (receptive_field_start_x, receptive_field_end_y - 1),
                     (receptive_field_end_x - 1, receptive_field_end_y - 1),
                     (receptive_field_end_x - 1, receptive_field_start_y)}

    print("[Receptive field ==> grad data]")
    i_line = 0
    for y in range(receptive_field_start_y, receptive_field_end_y):
        print("      ", end='')
        for x in range(receptive_field_start_x, receptive_field_end_x):
            if (x, y) in draw_as_blank:
                print("       ", end='')
            else:
                print("{:+01.3f},".format(sample_at(field, x, y)), end='')
        print(end_lines[i_line])
        i_line += 1


def compute_gradient_central_differences(field, x, y, verbose=False, use_replacement=False):
    if use_replacement:
        current_value = sample_at(field, x, y)
        live_y_minus_one = sample_at_replacement(field, current_value, x, y - 1)
        live_x_minus_one = sample_at_replacement(field, current_value, x - 1, y)
        live_y_plus_one = sample_at_replacement(field, current_value, x, y + 1)
        live_x_plus_one = sample_at_replacement(field, current_value, x + 1, y)
    else:
        live_y_minus_one = sample_at(field, x, y - 1)
        live_x_minus_one = sample_at(field, x - 1, y)
        live_y_plus_one = sample_at(field, x, y + 1)
        live_x_plus_one = sample_at(field, x + 1, y)
    x_grad = 0.5 * (live_x_plus_one - live_x_minus_one)
    y_grad = 0.5 * (live_y_plus_one - live_y_minus_one)

    if verbose:
        print_gradient_data(live_y_minus_one, live_x_minus_one, live_y_plus_one, live_x_plus_one,
                            sample_at(field, x, y))

    return np.array([x_grad, y_grad])


def compute_gradient_central_differences_smoothed(field, x, y, verbose=False):
    if 2 <= x < field.shape[1] - 2 and 2 <= y < field.shape[0] - 2:
        live_y_minus_one = convolve2d(field[y - 2:y + 1, x - 1:x + 2], gaussian_kernel3x3, mode='valid')[0, 0]
        live_x_minus_one = convolve2d(field[y - 1:y + 2, x - 2:x + 1], gaussian_kernel3x3, mode='valid')[0, 0]
        live_y_plus_one = convolve2d(field[y:y + 3, x - 1:x + 2], gaussian_kernel3x3, mode='valid')[0, 0]
        live_x_plus_one = convolve2d(field[y - 1:y + 2, x:x + 3], gaussian_kernel3x3, mode='valid')[0, 0]
    else:
        live_y_minus_one = sampling_convolve2d(field, gaussian_kernel3x3, x, y - 1)
        live_x_minus_one = sampling_convolve2d(field, gaussian_kernel3x3, x - 1, y)
        live_y_plus_one = sampling_convolve2d(field, gaussian_kernel3x3, x, y + 1)
        live_x_plus_one = sampling_convolve2d(field, gaussian_kernel3x3, x + 1, y)
    x_grad = 0.5 * (live_x_plus_one - live_x_minus_one)
    y_grad = 0.5 * (live_y_plus_one - live_y_minus_one)

    if verbose:
        if 1 <= x < field.shape[1] - 1 and 1 <= y < field.shape[0] - 1:
            central_value = convolve2d(field[y - 1:y + 2, x - 1:x + 2], gaussian_kernel3x3, mode='valid')[0, 0]
        else:
            central_value = sampling_convolve2d(field, gaussian_kernel3x3, x, y)
        print_gradient_data_3x3_receptive_field(field, x, y, live_y_minus_one, live_x_minus_one, live_y_plus_one,
                                                live_x_plus_one, central_value)

    return np.array([x_grad, y_grad])


def data_term_at_location_basic(warped_live_field, canonical_field, x, y, live_gradient_x, live_gradient_y):
    live_sdf = warped_live_field[y, x]
    canonical_sdf = canonical_field[y, x]

    diff = live_sdf - canonical_sdf

    if focus_coordinates_match(x, y):
        print("; Live - canonical: {:+01.4f} - {:+01.4f} = {:+01.4f}"
              .format(live_sdf, canonical_sdf, diff))
        compute_gradient_central_differences(warped_live_field, x, y, True, True)

    live_local_gradient = np.array([live_gradient_x[y, x], live_gradient_y[y, x]])

    scaling_factor = 10.0
    data_gradient = diff * live_local_gradient * scaling_factor

    local_energy_contribution = 0.5 * pow(diff, 2)

    return data_gradient, local_energy_contribution


def data_term_gradient(warped_live_field, canonical_field, scaling_factor=10.0):
    """
    Vectorized method to compute the data term gradient
    :param warped_live_field: current warped live SDF field
    :param canonical_field: canonical SDF field
    :param scaling_factor: scaling factor (usually determined by truncation point in SDF and narrow band
    width in voxels)
    :return: data gradient for each location as a matrix, data energy the entire grid summed up
    """
    diff = warped_live_field - canonical_field
    (live_gradient_x, live_gradient_y) = np.gradient(warped_live_field)
    data_gradient = diff * np.stack((live_gradient_x, live_gradient_y), axis=2) * scaling_factor
    data_energy = np.sum(0.5 * diff ** 2)
    return data_gradient, data_energy


def data_term_at_location_thresholded_fdm(warped_live_field, canonical_field, x, y, live_gradient_x, live_gradient_y):
    live_sdf = warped_live_field[y, x]
    canonical_sdf = canonical_field[y, x]

    diff = live_sdf - canonical_sdf

    if focus_coordinates_match(x, y):
        print("; Live - canonical: {:+01.4f} - {:+01.4f} = {:+01.4f}"
              .format(live_sdf, canonical_sdf, diff))
        compute_gradient_central_differences(warped_live_field, x, y, True, True)

    x_grad = live_gradient_x[y, x]
    if abs(x_grad) > 0.5:
        live_x_minus_one = sample_at(warped_live_field, x - 1, y)
        live_x_plus_one = sample_at(warped_live_field, x + 1, y)
        x_grad_forward = live_x_plus_one - live_sdf
        x_grad_backward = live_sdf - live_x_minus_one
        x_grad = x_grad_forward if abs(x_grad_forward) < abs(x_grad_backward) else x_grad_backward
        if abs(x_grad) > 0.5:
            x_grad = 0
    y_grad = live_gradient_y[y, x]
    if abs(y_grad) > 0.5:
        live_y_minus_one = sample_at(warped_live_field, x, y - 1)
        live_y_plus_one = sample_at(warped_live_field, x, y + 1)
        y_grad_forward = live_y_plus_one - live_sdf
        y_grad_backward = live_sdf - live_y_minus_one
        y_grad = y_grad_forward if abs(y_grad_forward) < abs(y_grad_backward) else y_grad_backward
        if abs(y_grad) > 0.5:
            y_grad = 0.0
    live_local_gradient = np.array([x_grad, y_grad])

    scaling_factor = 10.0
    data_gradient = diff * live_local_gradient * scaling_factor

    local_energy_contribution = 0.5 * pow(diff, 2)

    return data_gradient, local_energy_contribution


data_term_methods = {DataTermMethod.BASIC: data_term_at_location_basic,
                     DataTermMethod.THRESHOLDED_FDM: data_term_at_location_thresholded_fdm,
                     DataTermMethod.BASIC_CPP: lsfo.data_term_at_location}


def data_term_at_location(warped_live_field, canonical_field, x, y, live_gradient_x, live_gradient_y,
                          method=DataTermMethod.BASIC):
    return data_term_methods[method](warped_live_field, canonical_field, x, y, live_gradient_x, live_gradient_y)


def data_term_at_location_advanced_grad(warped_live_field, canonical_field, flag_field, x, y):
    live_sdf = warped_live_field[y, x]
    canonical_sdf = canonical_field[y, x]

    diff = live_sdf - canonical_sdf

    live_y_minus_one = sample_at(warped_live_field, x, y - 1)
    flag_y_minus_one = sample_flag_at(flag_field, x, y - 1)
    live_x_minus_one = sample_at(warped_live_field, x - 1, y)
    flag_x_minus_one = sample_flag_at(flag_field, x - 1, y)
    live_y_plus_one = sample_at(warped_live_field, x, y + 1)
    flag_y_plus_one = sample_flag_at(flag_field, x, y + 1)
    live_x_plus_one = sample_at(warped_live_field, x + 1, y)
    flag_x_plus_one = sample_flag_at(flag_field, x + 1, y)

    x_grad = 0.5 * (live_x_plus_one - live_x_minus_one)
    y_grad = 0.5 * (live_y_plus_one - live_y_minus_one)

    reduction_factor = 1.0
    special_used_x = False
    special_used_y = False

    if flag_y_plus_one == 0:
        if flag_y_minus_one == 0:
            y_grad = 0
        else:
            y_grad = reduction_factor * (live_sdf - live_y_minus_one)
        special_used_y = True
    elif flag_y_minus_one == 0:
        y_grad = reduction_factor * (live_y_plus_one - live_sdf)
        special_used_y = True

    if flag_x_plus_one == 0:
        if flag_x_minus_one == 0:
            x_grad = 0
        else:
            x_grad = reduction_factor * (live_sdf - live_x_minus_one)
        special_used_x = True
    elif flag_x_minus_one == 0:
        x_grad = reduction_factor * (live_x_plus_one - live_sdf)
        special_used_x = True

    # if live_y_plus_one == 1.0:
    #     if live_y_minus_one == 1.0:
    #         y_grad = 0
    #     else:
    #         y_grad = reduction_factor * (live_sdf - live_y_minus_one)
    #     special_used_y = True
    # elif live_y_minus_one == 1.0:
    #     y_grad = reduction_factor * (live_y_plus_one - live_sdf)
    #     special_used_y = True
    #
    # if live_x_plus_one == 1.0:
    #     if live_x_minus_one == 1.0:
    #         x_grad = 0
    #     else:
    #         x_grad = reduction_factor * (live_sdf - live_x_minus_one)
    #     special_used_x = True
    # elif live_x_minus_one == 1.0:
    #     x_grad = reduction_factor * (live_x_plus_one - live_sdf)
    #     special_used_x = True

    if focus_coordinates_match(x, y):
        print()
        print("[Grad data         ] Special fd x,y: ", BOLD_GREEN, special_used_x, ",", special_used_y, BOLD_BLUE,
              sep='')
        print("                     ", "      {:+01.3f}".format(live_y_minus_one), sep='')
        print("                     ",
              "{:+01.3f}{:+01.3f}{:+01.3f}".format(live_x_minus_one, live_sdf, live_x_plus_one), sep='')
        print("                     ", "      {:+01.3f}".format(live_y_plus_one), RESET, sep='')

    alternative_data = False
    if alternative_data:
        unscaled_warp_gradient_contribution = np.array([0.0, 0.0], dtype=np.float32)
        if abs(x_grad) > 1e-4:
            unscaled_warp_gradient_contribution[0] = diff / x_grad
        if abs(y_grad) > 1e-4:
            unscaled_warp_gradient_contribution[1] = diff / y_grad
    else:
        live_local_gradient = np.array([x_grad, y_grad])
        # scaling factor of 100 -- SDF values are 1/10 of the actual voxel distance, since we have 10 voxels in the
        # narrow band and the values go between 0 and 10. The diff is sampled at 1/10, so is the finite difference,
        # hence 10 * 10 is the correct factor
        unscaled_warp_gradient_contribution = diff * live_local_gradient * 100

    local_energy_contribution = 0.5 * pow(diff, 2)

    return unscaled_warp_gradient_contribution, local_energy_contribution
