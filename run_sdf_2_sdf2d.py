#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  Rigid alignment algorithm implementation based on SDF-2-SDF paper.
#  ================================================================

# common libs
import numpy as np

# local
from rigid_opt import sdf_2_sdf_visualizer as sdf2sdfv, sdf_2_sdf_optimizer2d as sdf2sdfo
from rigid_opt.sdf_generation import ImageBasedSingleFrameDataset
import utils.sampling as sampling
from calib.camera import DepthCamera

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


def main():
    canonical_frame_path = "../Data/Synthetic_Kenny_Circle/depth_000000.exr"
    live_frame_path = "../Data/Synthetic_Kenny_Circle/depth_000003.exr"
    image_pixel_row = 240

    intrinsic_matrix = np.matrix([[570.3999633789062, 0, 320],  # FX = 570.3999633789062 CX = 320.0
                                  [0, 570.3999633789062, 240],  # FY = 570.3999633789062 CY = 240.0
                                  [0, 0, 1]], dtype=np.float32)
    camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(480, 640),
                                                           intrinsic_matrix=intrinsic_matrix))
    field_size = 32
    offset = np.array([-16, -16, 104])
    data_to_use = ImageBasedSingleFrameDataset(
        canonical_frame_path,  # dataset from original sdf2sdf paper, reference frame
        live_frame_path,  # dataset from original sdf2sdf paper, current frame
        image_pixel_row, field_size, offset, camera
    )

    # depth_interpolation_method = tsdf.DepthInterpolationMethod.NONE
    out_path = "output/sdf2sdf"
    sampling.set_focus_coordinates(0, 0)
    narrow_band_width_voxels=1.
    iteration = 40
    optimizer = sdf2sdfo.Sdf2SdfOptimizer2d(
        verbosity_parameters=sdf2sdfo.Sdf2SdfOptimizer2d.VerbosityParameters(
            print_max_warp_update=True,
            print_iteration_energy=True
        ),
        visualization_parameters=sdf2sdfv.Sdf2SdfVisualizer.Parameters(
            out_path=out_path,
            save_initial_fields=True,
            save_final_fields=True,
            save_live_progression=True
        )
    )
    optimizer.optimize(data_to_use, narrow_band_width_voxels=narrow_band_width_voxels, iteration=iteration)

    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    main()
