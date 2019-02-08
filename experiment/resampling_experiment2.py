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
    data_to_use = data.PredefinedDatasetEnum.ZIGZAG064

    array_offset = np.array([-256, -256, 480], dtype=np.int32)
    field_size = np.array([512, 512, 512], dtype=np.int32)
    voxel_size = 0.004
    rig = DepthCameraRig.from_infinitam_format(
        "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag/inf_calib.txt")
    depth_camera = rig.depth_camera
    depth_interpolation_method = gen.DepthInterpolationMethod.EWA
    depth_image0 = cv2.imread(
        "/media/algomorph/Data/Reconstruction/synthetic_data/zigzag/input/depth_00064.png",
        cv2.IMREAD_UNCHANGED)
    max_depth = np.iinfo(np.uint16).max
    depth_image0[depth_image0 == 0] = max_depth
    field = \
        ewa.generate_3d_tsdf_field_from_depth_image_ewa_cpp(depth_image0,
                                                            depth_camera,
                                                            field_shape=field_size,
                                                            array_offset=array_offset,
                                                            voxel_size=voxel_size,
                                                            narrow_band_width_voxels=20)
    viz_image = ewa.generate_3d_tsdf_ewa_cpp_viz(depth_image=depth_image0,
                                                 camera=depth_camera,
                                                 field=field,
                                                 voxel_size=voxel_size,
                                                 array_offset=array_offset)
    cv2.imwrite("output/ewa_sampling_viz.png", viz_image)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
