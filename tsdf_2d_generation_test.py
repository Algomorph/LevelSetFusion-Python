#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 9/26/18.
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


import sys
from calib.camera import Camera
from calib.camerarig import DepthCameraRig
from tsdf_field_generation import generate_2d_tsdf_field_from_depth_image
from vizualization import process_cv_esc, sdf_field_to_image
import cv2

EXIT_CODE_SUCCESS = 0


def main():
    rig = DepthCameraRig.from_infinitam_format(
        "/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/inf_calib.txt")
    depth_camera = rig.depth_camera
    depth_image0 = cv2.imread("/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/input/depth_00000.png",
                              cv2.IMREAD_UNCHANGED)
    field0 = generate_2d_tsdf_field_from_depth_image(depth_image0, depth_camera, 200, default_value=1)
    depth_image1 = cv2.imread("/media/algomorph/Data/Reconstruction/synthetic_data/suzanne_away/input/depth_00001.png",
                              cv2.IMREAD_UNCHANGED)
    field1 = generate_2d_tsdf_field_from_depth_image(depth_image1, depth_camera, 200, default_value=1)
    # cv2.imshow("Field 0", sdf_field_to_image(field0))
    cv2.imwrite("test_field_0.png", sdf_field_to_image(field0))
    process_cv_esc()
    # cv2.imshow("Field 1", sdf_field_to_image(field1))
    cv2.imwrite("test_field_1.png", sdf_field_to_image(field1))
    process_cv_esc()

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
