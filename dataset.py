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

# Definitions of classes for meta-information about datasets and some convenience routines for data conversion

from enum import Enum

from abc import ABC, abstractmethod

import cv2
import numpy as np
from calib.camerarig import DepthCameraRig
import tsdf_field_generation as tsdf_gen


class DataToUse(Enum):
    GENEREATED2D = 0
    SIMPLE_TEST_CASE01 = 1

    SYNTHETIC3D_SUZANNE_AWAY = 10
    SYNTHETIC3D_SUZANNE_TWIST = 11

    SYNTHETIC3D_PLANE_AWAY = 20
    SYNTHETIC3D_PLANE_AWAY_512 = 21

    REAL3D_SNOOPY_SET01 = 100
    REAL3D_SNOOPY_SET02 = 101
    REAL3D_SNOOPY_SET03 = 102
    REAL3D_SNOOPY_SET04 = 103


class SingleFrameDataset(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_2d_sdf_fields(self, method=tsdf_gen.DepthInterpolationMethod.NONE):
        pass


class HardcodedSingleFrameDataset(SingleFrameDataset):
    def __init__(self, canonical_field, live_field):
        super(HardcodedSingleFrameDataset).__init__()
        self.field_size = canonical_field.shape[0]
        self.canonical_field = canonical_field
        self.live_field = live_field

    def generate_2d_sdf_fields(self, method=tsdf_gen.DepthInterpolationMethod.NONE):
        return self.live_field, self.canonical_field


class ImageBasedSingleFrameDataset(SingleFrameDataset):
    def __init__(self, calibration_file_path, first_frame_path, second_frame_path, image_pixel_row, field_size, offset):
        super(ImageBasedSingleFrameDataset).__init__()
        self.calibration_file_path = calibration_file_path
        self.first_frame_path = first_frame_path
        self.second_frame_path = second_frame_path
        self.image_pixel_row = image_pixel_row
        self.field_size = field_size
        self.offset = offset

    def generate_2d_sdf_fields(self, method=tsdf_gen.DepthInterpolationMethod.NONE):
        rig = DepthCameraRig.from_infinitam_format(self.calibration_file_path)
        depth_camera = rig.depth_camera
        depth_image0 = cv2.imread(self.first_frame_path, cv2.IMREAD_UNCHANGED)
        max_depth = np.iinfo(np.uint16).max
        depth_image0[depth_image0 == 0] = max_depth
        canonical_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image0, depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size, array_offset=self.offset,
                                                             depth_interpolation_method=method)
        depth_image1 = cv2.imread(self.second_frame_path, cv2.IMREAD_UNCHANGED)
        depth_image1[depth_image1 == 0] = max_depth
        live_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image1, depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size, array_offset=self.offset,
                                                             depth_interpolation_method=method)
        return live_field, canonical_field


class MaskedImageBasedSingleFrameDataset(SingleFrameDataset):
    def __init__(self, calibration_file_path, first_frame_path, first_mask_path, second_frame_path, second_mask_path,
                 image_pixel_row, field_size, offset):
        super(ImageBasedSingleFrameDataset).__init__()
        self.calibration_file_path = calibration_file_path
        self.first_frame_path = first_frame_path
        self.first_mask_path = first_mask_path
        self.second_frame_path = second_frame_path
        self.second_mask_path = second_mask_path
        self.image_pixel_row = image_pixel_row
        self.field_size = field_size
        self.offset = offset

    def generate_2d_sdf_fields(self, method=tsdf_gen.DepthInterpolationMethod.NONE):
        rig = DepthCameraRig.from_infinitam_format(self.calibration_file_path)
        depth_camera = rig.depth_camera
        depth_image0 = cv2.imread(self.first_frame_path, cv2.IMREAD_UNCHANGED)
        mask_image0 = cv2.imread(self.first_mask_path, cv2.IMREAD_UNCHANGED)
        max_depth = np.iinfo(np.uint16).max
        depth_image0[mask_image0 == 0] = max_depth
        depth_image0[depth_image0 == 0] = max_depth
        canonical_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image0, depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size, array_offset=self.offset,
                                                             depth_interpolation_method=method)
        depth_image1 = cv2.imread(self.second_frame_path, cv2.IMREAD_UNCHANGED)
        mask_image1 = cv2.imread(self.second_mask_path, cv2.IMREAD_UNCHANGED)
        depth_image1[mask_image1 == 0] = max_depth
        depth_image1[depth_image0 == 0] = max_depth
        live_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image1, depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size, array_offset=self.offset,
                                                             depth_interpolation_method=method)
        return live_field, canonical_field


datasets = {
    DataToUse.SYNTHETIC3D_SUZANNE_AWAY: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/inf_calib.txt",
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/input/depth_00001.png",
        200, 128, np.array([-64, -64, 0])
    ),
    DataToUse.SYNTHETIC3D_SUZANNE_TWIST: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_twist/inf_calib.txt",
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_twist/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_twist/input/depth_00001.png",
        200, 128, np.array([-64, -64, 64])
    ),
    DataToUse.REAL3D_SNOOPY_SET01: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000015.png",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000016.png",
        214, 128, np.array([-64, -64, 128])
    ),
    DataToUse.REAL3D_SNOOPY_SET02: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000064.png",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000065.png",
        214, 128, np.array([-64, -64, 128])
    ),
    DataToUse.REAL3D_SNOOPY_SET03: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000025.png",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000026.png",
        334, 128, np.array([-64, -64, 128])
    ),
    DataToUse.REAL3D_SNOOPY_SET04: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/snoopy_calib.txt",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000065.png",
        "/media/algomorph/Data/Reconstruction/real_data/KillingFusion Snoopy/frames/depth_000066.png",
        223, 128, np.array([-64, -64, 128])
    ),
    DataToUse.SYNTHETIC3D_PLANE_AWAY: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/inf_calib.txt",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00001.png",
        200, 128, np.array([-64, -64, 106])
    ),
    DataToUse.SYNTHETIC3D_PLANE_AWAY_512: ImageBasedSingleFrameDataset(
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/inf_calib.txt",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00000.png",
        "/media/algomorph/Data/Reconstruction/synthetic_data/plane_away/input/depth_00001.png",
        130, 512, np.array([-256, -256, 0])
    ),
    DataToUse.SIMPLE_TEST_CASE01: HardcodedSingleFrameDataset(
        np.array([[1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                  [1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                  [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                  [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32),
        np.array([[1., 1., 0.49999955, 0.42499956],
                  [1., 0.44999936, 0.34999937, 0.32499936],
                  [1., 0.35000065, 0.25000066, 0.22500065],
                  [1., 0.20000044, 0.15000044, 0.07500044]], dtype=np.float32))
}
