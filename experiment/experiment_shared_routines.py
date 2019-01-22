#  ================================================================
#  Created by Gregory Kramida on 11/29/18.
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

# contains some routines shared by the single-frame and multi-frame experiments

# stdlib
from enum import Enum
import re
import os
# libraries
import cv2
import numpy as np


# local

def is_unmasked_image_row_empty(path, ix_row):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return np.sum(image[ix_row]) == 0


def is_masked_image_row_empty(image_path, mask_path, ix_row):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    image[mask == 0] = 0
    return np.sum(image[ix_row]) == 0


def is_image_row_empty(image_path, mask_path, ix_row, check_masked):
    if check_masked:
        return is_masked_image_row_empty(image_path, mask_path, ix_row)
    else:
        return is_unmasked_image_row_empty(image_path, ix_row)


class FrameFilenameFormat(Enum):
    FIVE_DIGIT = 0
    SIX_DIGIT = 1


def check_frame_count_and_format(frames_path, turn_mask_checking_off=False):
    depth_five_digit_pattern = re.compile(r"^depth\_\d{5}[.]png$")
    depth_six_digit_pattern = re.compile(r"^depth\_\d{6}[.]png$")
    mask_five_digit_pattern = re.compile(r"^mask\_\d{5}[.]png$")
    mask_six_digit_pattern = re.compile(r"^mask\_\d{6}[.]png$")
    depth_five_digit_counter = 0
    depth_six_digit_counter = 0
    mask_five_digit_counter = 0
    mask_six_digit_counter = 0
    for filename in os.listdir(frames_path):
        if re.findall(depth_five_digit_pattern, filename):
            depth_five_digit_counter += 1
        elif re.findall(depth_six_digit_pattern, filename):
            depth_six_digit_counter += 1
        elif re.findall(mask_five_digit_pattern, filename):
            mask_five_digit_counter += 1
        elif re.findall(mask_six_digit_pattern, filename):
            mask_six_digit_counter += 1
    frame_count = max(depth_five_digit_counter, depth_six_digit_counter)
    filename_format = FrameFilenameFormat.FIVE_DIGIT if depth_five_digit_counter > depth_six_digit_counter \
        else FrameFilenameFormat.SIX_DIGIT
    use_masks = False
    if not turn_mask_checking_off and mask_five_digit_counter > 0 or mask_six_digit_counter > 0:
        use_masks = True
        if (filename_format == FrameFilenameFormat.FIVE_DIGIT and mask_five_digit_counter != depth_five_digit_counter)\
                or (filename_format == FrameFilenameFormat.SIX_DIGIT and
                    mask_six_digit_counter != depth_six_digit_counter):
            print("WARNING: Found some mask files, but could not establish correspondence with depth frames. "
                  "To be matched to depth filenames, mask filenames should use the same numbering format as the "
                  "depth filenames, there should be an equal number of both masks and depth frames, and the numbering"
                  "ranges should correspond.")
            use_masks = False
    return frame_count, filename_format, use_masks
