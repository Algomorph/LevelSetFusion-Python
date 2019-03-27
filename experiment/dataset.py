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

# stdlib
from enum import Enum
from abc import ABC, abstractmethod
import os.path

# libraries
import cv2
import numpy as np

# local
from calib.camerarig import DepthCameraRig
from tsdf import generation as tsdf_gen
import utils.path


class PredefinedDatasetEnum(Enum):
    GENEREATED2D = 0
    SIMPLE_TEST_CASE01 = 1

    SYNTHETIC3D_SUZANNE_AWAY = 10
    SYNTHETIC3D_SUZANNE_TWIST = 11

    SYNTHETIC3D_PLANE_AWAY = 20
    SYNTHETIC3D_PLANE_AWAY_512 = 21

    ZIGZAG001 = 30
    ZIGZAG064 = 33
    ZIGZAG124 = 35
    ZIGZAG248 = 39

    REAL3D_SNOOPY_SET00 = 100  # images available in tests/test_data
    REAL3D_SNOOPY_SET01 = 101
    REAL3D_SNOOPY_SET02 = 102
    REAL3D_SNOOPY_SET03 = 103
    REAL3D_SNOOPY_SET04 = 104
    REAL3D_SNOOPY_SET05 = 105
    REAL3D_SNOOPY_SET06 = 105


class FramePairDataset(ABC):
    def __init__(self, focus_coordinates=(-1, -1, -1)):
        self.focus_coordinates = focus_coordinates
        self.out_subpath = ""

    @abstractmethod
    def generate_2d_sdf_fields(self, method=tsdf_gen.GenerationMethod.BASIC, smoothing_coefficient=1.0):
        pass


class HardcodedFramePairDataset(FramePairDataset):
    def __init__(self, canonical_field, live_field):
        super(HardcodedFramePairDataset).__init__()
        self.field_size = canonical_field.shape[0]
        self.canonical_field = canonical_field
        self.live_field = live_field

    def generate_2d_sdf_fields(self, method=tsdf_gen.GenerationMethod.BASIC, smoothing_coefficient=1.0):
        return self.live_field, self.canonical_field


class ImageBasedFramePairDataset(FramePairDataset):
    def __init__(self, calibration_file_path, first_frame_path, second_frame_path, image_pixel_row, field_size, offset,
                 voxel_size=0.004, focus_coordinates=(-1, -1, -1)):
        super().__init__(focus_coordinates)
        self.calibration_file_path = calibration_file_path
        self.first_frame_path = first_frame_path
        self.second_frame_path = second_frame_path
        self.image_pixel_row = image_pixel_row
        self.field_size = field_size
        self.offset = offset
        self.voxel_size = voxel_size

    def generate_2d_sdf_canonical(self, method=tsdf_gen.GenerationMethod.BASIC, smoothing_coefficient=1.0):
        rig = DepthCameraRig.from_infinitam_format(self.calibration_file_path)
        depth_camera = rig.depth_camera
        depth_image0 = cv2.imread(self.first_frame_path, cv2.IMREAD_UNCHANGED)
        max_depth = np.iinfo(np.uint16).max
        depth_image0[depth_image0 == 0] = max_depth
        canonical_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image0, depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size, array_offset=self.offset,
                                                             generation_method=method,
                                                             voxel_size=self.voxel_size,
                                                             smoothing_coefficient=smoothing_coefficient)
        return canonical_field

    def generate_2d_sdf_live(self, method=tsdf_gen.GenerationMethod.BASIC, smoothing_coefficient=1.0):
        rig = DepthCameraRig.from_infinitam_format(self.calibration_file_path)
        depth_camera = rig.depth_camera
        depth_image1 = cv2.imread(self.second_frame_path, cv2.IMREAD_UNCHANGED)
        max_depth = np.iinfo(np.uint16).max
        depth_image1[depth_image1 == 0] = max_depth
        live_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image1, depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size, array_offset=self.offset,
                                                             generation_method=method,
                                                             voxel_size=self.voxel_size,
                                                             smoothing_coefficient=smoothing_coefficient)
        return live_field

    def generate_2d_sdf_fields(self, method=tsdf_gen.GenerationMethod.BASIC, smoothing_coefficient=1.0):
        live_field = self.generate_2d_sdf_live(method, smoothing_coefficient)
        canonical_field = self.generate_2d_sdf_canonical(method, smoothing_coefficient)
        return live_field, canonical_field


class MaskedImageBasedFramePairDataset(ImageBasedFramePairDataset):
    def __init__(self, calibration_file_path, first_frame_path, first_mask_path, second_frame_path, second_mask_path,
                 image_pixel_row, field_size, offset, voxel_size=0.004, focus_coordinates=(-1, -1, -1)):
        super().__init__(calibration_file_path, first_frame_path, second_frame_path,
                         image_pixel_row, field_size, offset,
                         voxel_size, focus_coordinates)
        self.first_mask_path = first_mask_path
        self.second_mask_path = second_mask_path

    def generate_2d_sdf_fields(self, method=tsdf_gen.GenerationMethod.BASIC, smoothing_coefficient=1.0):
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
                                                             generation_method=method,
                                                             smoothing_coefficient=smoothing_coefficient)
        depth_image1 = cv2.imread(self.second_frame_path, cv2.IMREAD_UNCHANGED)
        mask_image1 = cv2.imread(self.second_mask_path, cv2.IMREAD_UNCHANGED)
        depth_image1[mask_image1 == 0] = max_depth
        depth_image1[depth_image0 == 0] = max_depth
        live_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image1, depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size, array_offset=self.offset,
                                                             generation_method=method,
                                                             smoothing_coefficient=smoothing_coefficient)
        return live_field, canonical_field


datasets = {
    PredefinedDatasetEnum.ZIGZAG001: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00000.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00001.png"),
        200, 512, np.array([-256, -256, 640])
    ),
    PredefinedDatasetEnum.ZIGZAG064: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00064.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00065.png"),
        200, 512, np.array([-256, -256, 480])
    ),
    PredefinedDatasetEnum.ZIGZAG124: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00124.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00125.png"),
        200, 512, np.array([-256, -256, 360])
    ),
    PredefinedDatasetEnum.ZIGZAG248: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00248.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/zigzag/input/depth_00249.png"),
        200, 512, np.array([-256, -256, 256])
    ),
    PredefinedDatasetEnum.SYNTHETIC3D_SUZANNE_AWAY: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/suzanne_away/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(),
                     "synthetic_data/suzanne_away/input/depth_00000.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(),
                     "synthetic_data/suzanne_away/input/depth_00001.png"),
        200, 128, np.array([-64, -64, 0])
    ),
    PredefinedDatasetEnum.SYNTHETIC3D_SUZANNE_TWIST: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/suzanne_twist/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(),
                     "synthetic_data/suzanne_twist/input/depth_00000.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(),
                     "synthetic_data/suzanne_twist/input/depth_00010.png"),
        200, 128, np.array([-64, -64, 64])
    ),
    PredefinedDatasetEnum.REAL3D_SNOOPY_SET00: MaskedImageBasedFramePairDataset(
        utils.path.get_test_data_path("test_data/snoopy_calib.txt"),
        utils.path.get_test_data_path("test_data/snoopy_depth_000050.png"),
        utils.path.get_test_data_path("test_data/snoopy_omask_000050.png"),
        utils.path.get_test_data_path("test_data/snoopy_depth_000051.png"),
        utils.path.get_test_data_path("test_data/snoopy_omask_000051.png"),
        300, 128, np.array([-64, -64, 128])
    ),
    PredefinedDatasetEnum.REAL3D_SNOOPY_SET01: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/snoopy_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000015.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000016.png"),
        214, 128, np.array([-64, -64, 128])
    ),
    PredefinedDatasetEnum.REAL3D_SNOOPY_SET02: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/snoopy_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000064.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000065.png"),
        214, 128, np.array([-64, -64, 128])
    ),
    PredefinedDatasetEnum.REAL3D_SNOOPY_SET03: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/snoopy_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000025.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000026.png"),
        334, 128, np.array([-64, -64, 128])
    ),
    PredefinedDatasetEnum.REAL3D_SNOOPY_SET04: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/snoopy_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000065.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000066.png"),
        223, 128, np.array([-64, -64, 128])
    ),
    PredefinedDatasetEnum.REAL3D_SNOOPY_SET05: MaskedImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/snoopy_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000105.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/omask_000105.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000106.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/omask_000106.png"),
        355, 128, np.array([-64, -64, 128])
    ),
    PredefinedDatasetEnum.REAL3D_SNOOPY_SET06: MaskedImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/snoopy_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000650.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/omask_000650.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/depth_000651.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "real_data/snoopy/frames/omask_000651.png"),
        387, 128, np.array([-64, -64, 128])
    ),
    PredefinedDatasetEnum.SYNTHETIC3D_PLANE_AWAY: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/plane_away/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/plane_away/input/depth_00000.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/plane_away/input/depth_00001.png"),
        200, 128, np.array([-64, -64, 106])
    ),
    PredefinedDatasetEnum.SYNTHETIC3D_PLANE_AWAY_512: ImageBasedFramePairDataset(
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/plane_away/inf_calib.txt"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/plane_away/input/depth_00000.png"),
        os.path.join(utils.path.get_reconstruction_data_directory(), "synthetic_data/plane_away/input/depth_00001.png"),
        130, 512, np.array([-256, -256, 0])
    ),
    PredefinedDatasetEnum.SIMPLE_TEST_CASE01: HardcodedFramePairDataset(
        np.array([[1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                  [1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                  [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                  [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32),
        np.array([[1., 1., 0.49999955, 0.42499956],
                  [1., 0.44999936, 0.34999937, 0.32499936],
                  [1., 0.35000065, 0.25000066, 0.22500065],
                  [1., 0.20000044, 0.15000044, 0.07500044]], dtype=np.float32))
}
