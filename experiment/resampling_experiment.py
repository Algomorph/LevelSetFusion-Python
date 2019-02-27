#  ================================================================
#  Created by Gregory Kramida on 1/22/19.
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

# stdlib
import sys

# libs
import numpy as np

# local
import tsdf.ewa as ewa
import tsdf.generation as gen
import experiment.dataset as data
import utils.visualization as viz

# =========
import cv2
import matplotlib.pyplot as plt

from calib.camerarig import DepthCameraRig

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def main():
    save_profile = False
    fraction_field = False

    if fraction_field:
        image_path = "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag/depth/depth_00064.png"
    else:
        image_path = "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag2/input/depth_00108.png"

    # depth_interpolation_method = gen.DepthInterpolationMethod.NONE
    # depth_interpolation_method = gen.DepthInterpolationMethod.EWA
    depth_interpolation_method = gen.DepthInterpolationMethod.EWA_CPP

    if save_profile:
        im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        sl = im[200]
        plt.figure(figsize=(40, 30))
        plt.plot(sl, "k")
        filename = "../output/image_slice_plot.png"
        plt.savefig(filename)
        plt.clf()
        plt.close("all")
    else:

        if fraction_field:
            voxel_size = 0.004
            field_size = 16
            rig = DepthCameraRig.from_infinitam_format(
                "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag/inf_calib.txt")
            depth_camera = rig.depth_camera

            depth_image0 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            max_depth = np.iinfo(np.uint16).max
            depth_image0[depth_image0 == 0] = max_depth
            field = \
                gen.generate_2d_tsdf_field_from_depth_image(depth_image0, depth_camera, 200,
                                                            field_size=field_size,
                                                            array_offset=np.array([94, -256, 804]),
                                                            depth_interpolation_method=depth_interpolation_method,
                                                            voxel_size=voxel_size)
            print(repr(field))

        else:
            voxel_size = 0.004
            field_size = 512
            rig = DepthCameraRig.from_infinitam_format(
                "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag/inf_calib.txt")
            depth_camera = rig.depth_camera

            depth_image0 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            max_depth = np.iinfo(np.uint16).max
            depth_image0[depth_image0 == 0] = max_depth
            # z_offset = 480 # zigzag - 64
            z_offset = 0  # zigzag2 - 108
            field = \
                gen.generate_2d_tsdf_field_from_depth_image(depth_image0, depth_camera, 200,
                                                            field_size=field_size,
                                                            array_offset=np.array([-256, -256, z_offset]),
                                                            depth_interpolation_method=depth_interpolation_method,
                                                            voxel_size=voxel_size)
            print(field[103:119, 210:226].shape)

        viz.visualize_field(field, view_scaling_factor=2)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
