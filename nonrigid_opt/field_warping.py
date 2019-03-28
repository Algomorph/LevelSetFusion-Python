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
import math
from utils.point2d import Point2d
import utils.sampling as sampling
from utils.tsdf_set_routines import value_outside_narrow_band
from utils.printing import BOLD_YELLOW, BOLD_GREEN, RESET


def print_interpolation_data(metainfo, original_live_sdf, new_value):
    value00, value01, value10, value11, ratios, inverse_ratios = \
        metainfo.value00, metainfo.value01, metainfo.value10, metainfo.value11, metainfo.ratios, metainfo.inverse_ratios
    print("[Interpolation data] ", BOLD_YELLOW,
          "{:+03.3f}*{:03.3f}, {:+03.3f}*{:03.3f}".format(value00, inverse_ratios.y * inverse_ratios.x,
                                                          value10, inverse_ratios.y * ratios.x, ),
          RESET, " original value: ", BOLD_GREEN,
          original_live_sdf, RESET, sep='')
    print("                     ", BOLD_YELLOW,
          "{:+03.3f}*{:03.3f}, {:+03.3f}*{:03.3f}".format(value01, ratios.y * inverse_ratios.x,
                                                          value11, ratios.y * ratios.x),
          RESET, " final value: ", BOLD_GREEN,
          new_value, RESET, sep='')


def get_and_print_interpolation_data(canonical_field, warped_live_field, warp_field, x, y, band_union_only=False,
                                     known_values_only=False, substitute_original=False):
    # TODO: use in interpolation function (don't forget the component fields and the updates) to avoid DRY violation
    original_live_sdf = warped_live_field[y, x]
    original_live_sdf = warped_live_field[y, x]
    if band_union_only:
        canonical_sdf = canonical_field[y, x]
        if value_outside_narrow_band(original_live_sdf) and value_outside_narrow_band(canonical_sdf):
            return
    if known_values_only:
        if original_live_sdf == 1.0:
            return

    warped_location = Point2d(x, y) + Point2d(coordinates=warp_field[y, x])

    if substitute_original:
        new_value, metainfo = sampling.bilinear_sample_at_replacement_metainfo(warped_live_field, point=warped_location,
                                                                               replacement=original_live_sdf)
    else:
        new_value, metainfo = sampling.bilinear_sample_at_metainfo(warped_live_field, point=warped_location)

    if 1.0 - abs(new_value) < 1e-6:
        new_value = np.sign(new_value)

    print_interpolation_data(metainfo, original_live_sdf, new_value)


def warp_field(field, vector_field):
    """
    - Accepts a scalar field and a vector field [supposedly of the same dimensions & size -- not checked].
    - Creates a new scalar field of the same dimensions.
    - For each location of the vector field, performs a bilinear lookup in the scalar field.
    - If a vector is pointing outside of the bounds of the input fields, uses the value "1" during
    the bilinear lookup for any "out-of-bounds" spots.
    - Stores the results of each lookup in the corresponding location of the new scalar field
    :param field: the scalar field containing source values
    :param vector_field: 2d vector field to use for bilinear lookups
    :return: the resulting scalar field
    """
    warped_field = np.ones_like(field)
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            warped_location = Point2d(x, y) + Point2d(coordinates=vector_field[y, x])
            new_value = sampling.bilinear_sample_at(field, point=warped_location)
            warped_field[y, x] = new_value
    return warped_field


def warp_field_replacement(field, warps, replacement):
    """
    - Accepts a scalar field and a vector field [supposedly of the same dimensions & size -- not checked].
    - Creates a new scalar field of the same dimensions
    - For each location of the vector field, performs a bilinear lookup in the scalar field
    - If a vector is pointing outside of the bounds of the input fields, uses the replacement during
    the interpolation process for any "out-of-bounds" spots.
    - Stores the results of each lookup in the corresponding location of the new scalar field
    :param field: the scalar field containing source values
    :param warps: 2d vector field to use for bilinear lookups
    :param replacement: value to use when warp points outside the span of the field
    :return: the resulting scalar field
    """
    warped_field = np.ones_like(field)
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            warped_location = Point2d(x, y) + Point2d(coordinates=warps[y, x])
            new_value = sampling.bilinear_sample_at_replacement(field,
                                                                point=warped_location,
                                                                replacement=replacement)
            warped_field[y, x] = new_value
    return warped_field


def warp_field_advanced(canonical_field, warped_live_field, warp_field, gradient_field, band_union_only=False,
                        known_values_only=False, substitute_original=False,
                        data_gradient_field=None, smoothing_gradient_field=None):
    field_size = warp_field.shape[0]
    new_warped_live_field = np.ones_like(warped_live_field)
    for y in range(field_size):
        for x in range(field_size):
            original_live_sdf = warped_live_field[y, x]
            if band_union_only:
                canonical_sdf = canonical_field[y, x]
                if value_outside_narrow_band(original_live_sdf) and value_outside_narrow_band(canonical_sdf):
                    new_warped_live_field[y, x] = original_live_sdf
                    continue
            if known_values_only:
                if original_live_sdf == 1.0:
                    new_warped_live_field[y, x] = original_live_sdf
                    continue

            warped_location = Point2d(x, y) + Point2d(coordinates=warp_field[y, x])

            if substitute_original:
                new_value, metainfo = sampling.bilinear_sample_at_replacement_metainfo(warped_live_field,
                                                                                       point=warped_location,
                                                                                       replacement=original_live_sdf)
            else:
                new_value, metainfo = sampling.bilinear_sample_at_metainfo(warped_live_field, point=warped_location)

            if 1.0 - abs(new_value) < 1e-6:
                new_value = np.sign(new_value)
                warp_field[y, x] = 0.0
                gradient_field[y, x] = 0.0
                if data_gradient_field is not None:
                    data_gradient_field[y, x] = 0.0
                if smoothing_gradient_field is not None:
                    smoothing_gradient_field[y, x] = 0.0

            if sampling.focus_coordinates_match(x, y):
                print_interpolation_data(metainfo, original_live_sdf, new_value)
            new_warped_live_field[y, x] = new_value
    return new_warped_live_field


def warp_field_with_with_flag_info(warped_live_field, warp_field, update_field, flag_field):
    field_size = warp_field.shape[0]
    new_warped_live_field = np.ones_like(warped_live_field)
    for y in range(field_size):
        for x in range(field_size):
            warped_location = Point2d(x, y) + Point2d(coordinates=warp_field[y, x])
            base_point = Point2d(math.floor(warped_location.x), math.floor(warped_location.y))
            ratios = warped_location - base_point
            inverse_ratios = Point2d(1.0, 1.0) - ratios
            original_value = warped_live_field[y, x]
            value00 = sampling.sample_at(warped_live_field, point=base_point)
            flag00 = sampling.sample_flag_at(flag_field, point=base_point)
            used_replacement = False
            if flag00 == 0:
                value00 = original_value
                used_replacement = True
            value01 = sampling.sample_at(warped_live_field, point=base_point + Point2d(0, 1))
            flag01 = sampling.sample_flag_at(flag_field, point=base_point + Point2d(0, 1))
            if flag01 == 0:
                value01 = original_value
                used_replacement = True
            value10 = sampling.sample_at(warped_live_field, point=base_point + Point2d(1, 0))
            flag10 = sampling.sample_flag_at(flag_field, point=base_point + Point2d(1, 0))
            if flag10 == 0:
                value10 = original_value
                used_replacement = True
            value11 = sampling.sample_at(warped_live_field, point=base_point + Point2d(1, 1))
            flag11 = sampling.sample_flag_at(flag_field, point=base_point + Point2d(1, 1))
            if flag11 == 0:
                value11 = original_value
                used_replacement = True

            interpolated_value0 = value00 * inverse_ratios.y + value01 * ratios.y
            interpolated_value1 = value10 * inverse_ratios.y + value11 * ratios.y
            interpolated_value = interpolated_value0 * inverse_ratios.x + interpolated_value1 * ratios.x
            if 1.0 - abs(interpolated_value) < 1e-3:
                # if 1.0 - abs(interpolated_value) < 0.05:
                interpolated_value = np.sign(interpolated_value)
                warp_field[y, x] = 0.0
                update_field[y, x] = 0.0

            if sampling.focus_coordinates_match(x, y):
                print("[Interpolation data] ", BOLD_YELLOW,
                      "{:+03.3f}*{:03.3f}, {:+03.3f}*{:03.3f}".format(value00, inverse_ratios.y * inverse_ratios.x,
                                                                      value10, inverse_ratios.y * ratios.x, ),
                      RESET, sep='')
                print("                     ", BOLD_YELLOW,
                      "{:+03.3f}*{:03.3f}, {:+03.3f}*{:03.3f}".format(value01, ratios.y * inverse_ratios.x,
                                                                      value11, ratios.y * ratios.x),
                      RESET, " used replacement:", BOLD_GREEN, used_replacement, RESET, " final value: ", BOLD_GREEN,
                      interpolated_value, RESET, sep='')
            new_warped_live_field[y, x] = interpolated_value
    np.copyto(warped_live_field, new_warped_live_field)
