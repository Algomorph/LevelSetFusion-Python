#  ================================================================
#  Created by Gregory Kramida on 10/17/18.
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


def set_zeros_for_values_outside_narrow_band_union(warped_live_field, canonical_field, target_field):
    """
    nullifies the effects outside of the narrow band
    :param warped_live_field: live SDF
    :param canonical_field: canonical SDF
    :param target_field: target field or iterable of target fields
    """
    truncated = np.bitwise_and(np.abs(warped_live_field) == 1.0, np.abs(canonical_field) == 1.0)
    target_field[truncated] = 0.0  # nullifies the effects outside of the narrow band


def set_zeros_for_values_outside_narrow_band_union_multitarget(warped_live_field, canonical_field, targets):
    """
    nullifies the effects outside of the narrow band
    :param warped_live_field: live SDF
    :param canonical_field: canonical SDF
    :param targets: iterable of target fields
    """
    truncated = np.bitwise_and(np.abs(warped_live_field) == 1.0, np.abs(canonical_field) == 1.0)
    for target_field in targets:
        target_field[truncated] = 0.0  # nullifies the effects outside of the narrow band


def value_outside_narrow_band(sdf_value):
    return sdf_value == 1.0 or sdf_value == -1.0  # or sdf_value == 0.0


def voxel_is_outside_narrow_band_union(live_field, canonical_field, x, y):
    live_sdf = live_field[y, x]
    canonical_sdf = canonical_field[y, x]

    live_is_truncated = value_outside_narrow_band(live_sdf)
    canonical_is_truncated = value_outside_narrow_band(canonical_sdf)
    return live_is_truncated and canonical_is_truncated
