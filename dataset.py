#  ================================================================
#  Created by Gregory Kramida on 10/5/18.
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

import cv2
import numpy as np
from calib.camerarig import DepthCameraRig
from tsdf_field_generation import generate_2d_tsdf_field_from_depth_image


class DataToUse(Enum):
    SYNTHETIC3D_SUZANNE_AWAY = 1
    GENEREATED2D = 2
    REAL3D_SNOOPY_SET01 = 3
    REAL3D_SNOOPY_SET02 = 4
    REAL3D_SNOOPY_SET03 = 5
    SYNTHETIC3D_PLANE_AWAY = 10
    SYNTHETIC3D_PLANE_AWAY_512 = 11


class Dataset:
    def __init__(self, calibration_file_path, first_frame_path, second_frame_path, image_pixel_row, field_size, offset):
        self.calibration_file_path = calibration_file_path
        self.first_frame_path = first_frame_path
        self.second_frame_path = second_frame_path
        self.image_pixel_row = image_pixel_row
        self.field_size = field_size
        self.offset = offset

    def generate_2d_sdf_fields(self, default_value=1):
        rig = DepthCameraRig.from_infinitam_format(self.calibration_file_path)
        depth_camera = rig.depth_camera
        depth_image0 = cv2.imread(self.first_frame_path, cv2.IMREAD_UNCHANGED)
        canonical_field = generate_2d_tsdf_field_from_depth_image(depth_image0, depth_camera, self.image_pixel_row,
                                                                  default_value=default_value,
                                                                  field_size=self.field_size,
                                                                  array_offset=self.offset)
        depth_image1 = cv2.imread(self.second_frame_path, cv2.IMREAD_UNCHANGED)
        live_field = generate_2d_tsdf_field_from_depth_image(depth_image1, depth_camera, self.image_pixel_row,
                                                             default_value=default_value, field_size=self.field_size,
                                                             array_offset=self.offset)
        return live_field, canonical_field


datasets = {
    DataToUse.SYNTHETIC3D_SUZANNE_AWAY: Dataset(
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/inf_calib.txt",
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/input/depth_00001.png",
        200, 128, np.array([-64, -64, 0])
    ),
    DataToUse.REAL3D_SNOOPY_SET01: Dataset(
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000015.png",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000016.png",
        214, 128, np.array([-64, -64, 128])
    ),
    DataToUse.REAL3D_SNOOPY_SET02: Dataset(
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000064.png",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000065.png",
        214, 128, np.array([-64, -64, 128])
    ),
    DataToUse.REAL3D_SNOOPY_SET03: Dataset(
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000025.png",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000026.png",
        334, 128, np.array([-64, -64, 128])
    ),
    DataToUse.SYNTHETIC3D_PLANE_AWAY: Dataset(
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/inf_calib.txt",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00001.png",
        200, 128, np.array([-64, -64, 106])
    ),
    DataToUse.SYNTHETIC3D_PLANE_AWAY_512: Dataset(
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/inf_calib.txt",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00001.png",
        130, 512, np.array([-256, -256, 0])
    )
}
