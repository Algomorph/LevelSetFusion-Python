#  ================================================================
#  Created by Gregory Kramida on 3/5/19.
#  Copyright (c) 2019 Gregory Kramida
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

class GenerationMethod:
    NONE = 0
    BILINEAR_IMAGE = 1
    BILINEAR_TSDF = 2
    EWA_IMAGE = 3
    EWA_IMAGE_CPP = 4
    EWA_TSDF = 5
    EWA_TSDF_CPP = 6
    EWA_TSDF_INCLUSIVE = 7
    EWA_TSDF_INCLUSIVE_CPP = 8


def compute_tsdf_value(signed_distance, narrow_band_half_width):
    """
    Compute TSDF value as narrow band width fraction based on provided SDF and narrow band half-width
    :param signed_distance: signed distance in metric units
    :param narrow_band_half_width: half-width of the narrow band in metric units
    :return: result TSDF value
    """
    if signed_distance < -narrow_band_half_width:
        tsdf_value = -1.0
    elif signed_distance > narrow_band_half_width:
        tsdf_value = 1.0
    else:
        tsdf_value = signed_distance / narrow_band_half_width
    return tsdf_value
