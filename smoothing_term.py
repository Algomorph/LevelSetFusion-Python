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
from enum import Enum
import numpy as np
from utils.printing import *
import scipy.ndimage
import scipy

from utils.sampling import focus_coordinates_match, sample_warp_replace_if_zero, sample_warp
from utils.tsdf_set_routines import set_zeros_for_values_outside_narrow_band_union_multitarget, \
    value_outside_narrow_band, voxel_is_outside_narrow_band_union, set_zeros_for_values_outside_narrow_band_union


class SmoothingTermMethod(Enum):
    TIKHONOV = 0
    KILLING = 1


# region ==================================== PRINTING ROUTINES ========================================================
def print_smoothing_term_data(warp_y_minus_one, warp_x_minus_one, warp_y_plus_one, warp_x_plus_one, warp):
    print()
    print("[Warp data         ]", BOLD_LIGHT_CYAN, sep='')
    print("                     ", "                 [{:+01.4f},{:+01.4f}]"
          .format(warp_y_minus_one[0], warp_y_minus_one[1]), sep='')
    print("                     ",
          "[{:+01.4f},{:+01.4f}][{:+01.4f},{:+01.4f}][{:+01.4f},{:+01.4f}]"
          .format(warp_x_minus_one[0], warp_x_minus_one[1], warp[0], warp[1], warp_x_plus_one[0], warp_x_plus_one[1]),
          sep='')
    print("                     ", "                 [{:+01.4f},{:+01.4f}]"
          .format(warp_y_plus_one[0], warp_y_plus_one[1]), RESET, sep='')


# endregion


# region ==================================== LOCAL GRADIENTS ==========================================================
def compute_local_smoothing_term_gradient_killing(warp_field, x, y, ignore_if_zero=False, copy_if_zero=True,
                                                  isomorphic_enforcement_factor=0.1):
    warp = warp_field[y, x]

    if copy_if_zero:
        warp_x_plus_one = sample_warp_replace_if_zero(warp_field, x + 1, y, warp)
        warp_x_minus_one = sample_warp_replace_if_zero(warp_field, x - 1, y, warp)
        warp_y_plus_one = sample_warp_replace_if_zero(warp_field, x, y + 1, warp)
        warp_y_minus_one = sample_warp_replace_if_zero(warp_field, x, y - 1, warp)
    else:
        warp_x_plus_one = sample_warp(warp_field, x + 1, y, warp)
        warp_x_minus_one = sample_warp(warp_field, x - 1, y, warp)
        warp_y_plus_one = sample_warp(warp_field, x, y + 1, warp)
        warp_y_minus_one = sample_warp(warp_field, x, y - 1, warp)

    warp_gradient_x = 0.5 * (warp_x_plus_one - warp_x_minus_one)
    warp_gradient_y = 0.5 * (warp_y_plus_one - warp_y_minus_one)

    warp_gradient_xx = warp_x_plus_one - 2 * warp + warp_x_minus_one  # [u_xx, v_xx]
    warp_gradient_yy = warp_y_plus_one - 2 * warp + warp_y_plus_one  # [u_yy, v_yy]

    if copy_if_zero:
        warp_x_plus_one_y_plus_one = sample_warp_replace_if_zero(warp_field, x + 1, y + 1, warp)
        warp_x_minus_one_y_plus_one = sample_warp_replace_if_zero(warp_field, x - 1, y + 1, warp)
        warp_x_plus_one_y_minus_one = sample_warp_replace_if_zero(warp_field, x + 1, y - 1, warp)
        warp_x_minus_one_y_minus_one = sample_warp_replace_if_zero(warp_field, x - 1, y - 1, warp)
    else:
        warp_x_plus_one_y_plus_one = sample_warp(warp_field, x + 1, y + 1, warp)
        warp_x_minus_one_y_plus_one = sample_warp(warp_field, x - 1, y + 1, warp)
        warp_x_plus_one_y_minus_one = sample_warp(warp_field, x + 1, y - 1, warp)
        warp_x_minus_one_y_minus_one = sample_warp(warp_field, x - 1, y - 1, warp)

    # see http://www.iue.tuwien.ac.at/phd/heinzl/node27.html
    warp_gradient_xy = (warp_x_plus_one_y_plus_one - warp_x_plus_one_y_minus_one -
                        warp_x_minus_one_y_plus_one + warp_x_minus_one_y_minus_one) / 4.0  # [u_xy, v_xy]

    # -2((1+lambda)u_xx + u_yy + (lambda)v_xy
    # -2((1+lambda)v_yy + v_xx + (lambda)u_xy
    lambda_ = isomorphic_enforcement_factor
    smoothing_gradient = np.array([
        -2 * (1 + lambda_) * warp_gradient_xx[0] + warp_gradient_yy[0] + lambda_ * warp_gradient_xy[1],
        -2 * (1 + lambda_) * warp_gradient_xx[1] + warp_gradient_yy[1] + lambda_ * warp_gradient_xy[0],
    ])
    vec_jacobian = np.hstack((warp_gradient_x, warp_gradient_y))
    vec_jacobian_transpose = \
        np.array([warp_gradient_x[0], warp_gradient_y[0], warp_gradient_x[1], warp_gradient_y[1]])

    local_energy_contribution = vec_jacobian.dot(vec_jacobian) + isomorphic_enforcement_factor * \
                                (vec_jacobian_transpose.dot(vec_jacobian))

    return smoothing_gradient, local_energy_contribution


def compute_local_smoothing_term_gradient_tikhonov(warp_field, x, y, ignore_if_zero=False, copy_if_zero=True,
                                                   isomorphic_enforcement_factor=0.1):
    # 1D discrete laplacian using finite-differences
    warp = warp_field[y, x]

    if ignore_if_zero and \
            ((x != warp_field.shape[1] - 1 and np.linalg.norm(warp_field[y, x + 1] == 0.0))
             or (x != 0 and np.linalg.norm(warp_field[y, x - 1] == 0.0))
             or (y != warp_field.shape[0] - 1 and np.linalg.norm(warp_field[y + 1, x] == 0.0))
             or (y != 0 and np.linalg.norm(warp_field[y - 1, x] == 0.0))):
        return np.array([0.0, 0.0], dtype=np.float32), 0.0

    if copy_if_zero:
        warp_x_plus_one = sample_warp_replace_if_zero(warp_field, x + 1, y, warp)
        warp_x_minus_one = sample_warp_replace_if_zero(warp_field, x - 1, y, warp)
        warp_y_plus_one = sample_warp_replace_if_zero(warp_field, x, y + 1, warp)
        warp_y_minus_one = sample_warp_replace_if_zero(warp_field, x, y - 1, warp)
    else:
        warp_x_plus_one = sample_warp(warp_field, x + 1, y, warp)
        warp_x_minus_one = sample_warp(warp_field, x - 1, y, warp)
        warp_y_plus_one = sample_warp(warp_field, x, y + 1, warp)
        warp_y_minus_one = sample_warp(warp_field, x, y - 1, warp)

    if focus_coordinates_match(x, y):
        print_smoothing_term_data(warp_y_minus_one, warp_x_minus_one, warp_y_plus_one, warp_x_plus_one, warp)

    scaling_factor = 1.0

    smoothing_gradient = -scaling_factor * (
            warp_x_plus_one + warp_y_plus_one - 4 * warp + warp_x_minus_one + warp_y_minus_one)

    warp_gradient_x = 0.5 * (warp_x_plus_one - warp_x_minus_one) * scaling_factor
    warp_gradient_y = 0.5 * (warp_y_plus_one - warp_y_minus_one) * scaling_factor

    local_energy_contribution = 0.5 * (
            warp_gradient_x.dot(warp_gradient_x) + warp_gradient_y.dot(warp_gradient_y))
    return smoothing_gradient, local_energy_contribution


smoothing_term_methods = {SmoothingTermMethod.KILLING: compute_local_smoothing_term_gradient_killing,
                          SmoothingTermMethod.TIKHONOV: compute_local_smoothing_term_gradient_tikhonov}


def compute_local_smoothing_term_gradient(warp_field, x, y, ignore_if_zero=False,
                                          copy_if_zero=True, method=SmoothingTermMethod.TIKHONOV,
                                          isomorphic_enforcement_factor=0.1):
    return smoothing_term_methods[method](warp_field, x, y, ignore_if_zero, copy_if_zero, isomorphic_enforcement_factor)


# endregion


def compute_smoothing_term_gradient_vectorized(warp_field):
    laplace_u = scipy.ndimage.laplace(warp_field[:, :, 0])
    laplace_v = scipy.ndimage.laplace(warp_field[:, :, 1])
    smoothing_gradient = -np.stack((laplace_u, laplace_v), axis=2)
    return smoothing_gradient


def compute_smoothing_term_energy(warp_field, warped_live_field=None, canonical_field=None, band_union_only=True):
    if band_union_only and (warped_live_field is None or canonical_field is None):
        raise ValueError(
            "To determine the narrow band union, warped_live_field and canonical_field should be defined."
            " Otherwise, please set the 'band_union_only argument' to 'False'")

    warp_gradient_u_x, warp_gradient_u_y = np.gradient(warp_field[:, :, 0])
    warp_gradient_v_x, warp_gradient_v_y = np.gradient(warp_field[:, :, 1])

    # if band_union_only:
    #     set_zeros_for_values_outside_narrow_band_union_multitarget(warped_live_field, canonical_field,
    #                                                                (warp_gradient_u_x, warp_gradient_u_y,
    #                                                                 warp_gradient_v_x, warp_gradient_v_y))

    gradient_aggregate = \
        warp_gradient_u_x ** 2 + warp_gradient_v_x ** 2 + warp_gradient_u_y ** 2 + warp_gradient_v_y ** 2
    if band_union_only:
        set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, gradient_aggregate)

    smoothing_energy = 0.5 * np.sum(gradient_aggregate)
    return smoothing_energy


def compute_smoothing_term_gradient_direct(warp_field, warped_live_field, canonical_field, band_union_only=True):
    """
    Computes the data gradient directly by traversing the 2D grid (live and canonical scalar fields) and computing the
     gradient separately at each location. Made mostly for testing the vectorized version.
    :param warped_live_field:
    :param canonical_field:
    :param live_gradient_x:
    :param live_gradient_y:
    :param band_union_only:
    :return:
    """
    smoothing_gradient_field = np.zeros((warped_live_field.shape[0], warped_live_field.shape[1], 2), dtype=np.float32)
    total_smoothing_energy = 0.0
    for y in range(0, warped_live_field.shape[0]):
        for x in range(0, warped_live_field.shape[1]):
            if band_union_only and voxel_is_outside_narrow_band_union(warped_live_field, canonical_field, x, y):
                continue
            smoothing_gradient, local_smoothing_energy = \
                compute_local_smoothing_term_gradient_tikhonov(warp_field, x, y, copy_if_zero=False)
            total_smoothing_energy += local_smoothing_energy
            smoothing_gradient_field[y, x] = smoothing_gradient
    return smoothing_gradient_field, total_smoothing_energy
