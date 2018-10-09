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
FOCUS_COORDINATES = (53, 87)


def sample_at(field, x=0, y=0, point=None):
    if point is not None:
        x = point.x
        y = point.y
    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        return 1
    return field[y, x]


def sample_flag_at(field, x=0, y=0, point=None):
    if point is not None:
        x = point.x
        y = point.y
    if x < 0.0 or x >= field.shape[1] or y < 0.0 or y >= field.shape[0]:
        return 0
    return field[y, x]


def sample_at_replacement(field, replacement, x=0, y=0, point=None):
    if point is not None:
        x = point.x
        y = point.y
    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        return replacement
    return field[y, x]


def is_outside_narrow_band(sdf_value):
    return sdf_value == 1.0 or sdf_value == -1.0  # or sdf_value == 0.0


def focus_coordinates_match(x, y):
    return x == FOCUS_COORDINATES[0] and y == FOCUS_COORDINATES[1]


def get_focus_coordinates():
    return FOCUS_COORDINATES


def sample_warp(warp_field, x, y, replacement):
    if x >= warp_field.shape[1] or y >= warp_field.shape[0] - 1 or x < 0 or y < 0:
        return replacement
    else:
        return warp_field[y, x]


def sample_warp_replace_if_zero(warp_field, x, y, replacement):
    if x >= warp_field.shape[1] or y >= warp_field.shape[0] - 1 or x < 0 or y < 0 or np.linalg.norm(
            warp_field[y, x]) == 0.0:
        return replacement
    else:
        return warp_field[y, x]