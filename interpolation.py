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
from utils.point import Point
from utils.sampling import sample_at, focus_coordinates_match, is_outside_narrow_band, sample_flag_at, sample_at_replacement
from utils.printing import BOLD_YELLOW, BOLD_GREEN, RESET


def interpolate_warped_live(canonical_field, warped_live_field, warp_field, gradient_field, band_union_only=False,
                            known_values_only=False,
                            data_gradient_field=None, smoothing_gradient_field=None, substitute_original=False):
    field_size = warp_field.shape[0]
    new_warped_live_field = np.ones_like(warped_live_field)
    for y in range(field_size):
        for x in range(field_size):
            original_live_sdf = warped_live_field[y, x]
            if band_union_only:
                canonical_sdf = canonical_field[y, x]
                if is_outside_narrow_band(original_live_sdf) and is_outside_narrow_band(canonical_sdf):
                    continue
            if known_values_only:
                if original_live_sdf == 1.0:
                    continue

            warped_location = Point(x, y) + Point(coordinates=warp_field[y, x])
            base_point = Point(math.floor(warped_location.x), math.floor(warped_location.y))
            ratios = warped_location - base_point
            inverse_ratios = Point(1.0, 1.0) - ratios

            if substitute_original:
                value00 = sample_at_replacement(warped_live_field, original_live_sdf, point=base_point)
                value01 = sample_at_replacement(warped_live_field, original_live_sdf, point=base_point + Point(0, 1))
                value10 = sample_at_replacement(warped_live_field, original_live_sdf, point=base_point + Point(1, 0))
                value11 = sample_at_replacement(warped_live_field, original_live_sdf, point=base_point + Point(1, 1))
            else:
                value00 = sample_at(warped_live_field, point=base_point)
                value01 = sample_at(warped_live_field, point=base_point + Point(0, 1))
                value10 = sample_at(warped_live_field, point=base_point + Point(1, 0))
                value11 = sample_at(warped_live_field, point=base_point + Point(1, 1))

            interpolated_value0 = value00 * inverse_ratios.y + value01 * ratios.y
            interpolated_value1 = value10 * inverse_ratios.y + value11 * ratios.y
            new_value = interpolated_value0 * inverse_ratios.x + interpolated_value1 * ratios.x
            if 1.0 - abs(new_value) < 1e-6:
                new_value = np.sign(new_value)
                warp_field[y, x] = 0.0
                gradient_field[y, x] = 0.0
                if data_gradient_field is not None:
                    data_gradient_field[y, x] = 0.0
                if smoothing_gradient_field is not None:
                    smoothing_gradient_field[y, x] = 0.0

            if focus_coordinates_match(x, y):
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
            new_warped_live_field[y, x] = new_value
    np.copyto(warped_live_field, new_warped_live_field)


def interpolate_warped_live_with_flag_info(warped_live_field, warp_field, update_field, flag_field):
    field_size = warp_field.shape[0]
    new_warped_live_field = np.ones_like(warped_live_field)
    for y in range(field_size):
        for x in range(field_size):
            warped_location = Point(x, y) + Point(coordinates=warp_field[y, x])
            base_point = Point(math.floor(warped_location.x), math.floor(warped_location.y))
            ratios = warped_location - base_point
            inverse_ratios = Point(1.0, 1.0) - ratios
            original_value = warped_live_field[y, x]
            value00 = sample_at(warped_live_field, point=base_point)
            flag00 = sample_flag_at(flag_field, point=base_point)
            used_replacement = False
            if flag00 == 0:
                value00 = original_value
                used_replacement = True
            value01 = sample_at(warped_live_field, point=base_point + Point(0, 1))
            flag01 = sample_flag_at(flag_field, point=base_point + Point(0, 1))
            if flag01 == 0:
                value01 = original_value
                used_replacement = True
            value10 = sample_at(warped_live_field, point=base_point + Point(1, 0))
            flag10 = sample_flag_at(flag_field, point=base_point + Point(1, 0))
            if flag10 == 0:
                value10 = original_value
                used_replacement = True
            value11 = sample_at(warped_live_field, point=base_point + Point(1, 1))
            flag11 = sample_flag_at(flag_field, point=base_point + Point(1, 1))
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

            if focus_coordinates_match(x, y):
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