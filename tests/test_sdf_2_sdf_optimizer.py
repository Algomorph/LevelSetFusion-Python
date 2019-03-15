from unittest import TestCase
import numpy as np
from calib.camera import DepthCamera
from rigid_opt import sdf_2_sdf_visualizer as sdf2sdfv, sdf_2_sdf_optimizer2d as sdf2sdfo
from rigid_opt.sdf_generation import ImageBasedSingleFrameDataset
import utils.sampling as sampling
import utils.path
import os.path


class MyTestCase(TestCase):

    def test_sdf_2_sdf_optimizer01(self):
        canonical_frame_path = utils.path.get_test_data_path("test_data/depth_000000.exr")
        live_frame_path = utils.path.get_test_data_path("test_data/depth_000003.exr")

        image_pixel_row = 240

        intrinsic_matrix = np.array([[570.3999633789062, 0, 320],  # FX = 570.3999633789062 CX = 320.0
                                      [0, 570.3999633789062, 240],  # FY = 570.3999633789062 CY = 240.0
                                      [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(480, 640),
                                                               intrinsic_matrix=intrinsic_matrix))
        field_size = 32
        offset = np.array([-16, -16, 93.4375])

        data_to_use = ImageBasedSingleFrameDataset(
            canonical_frame_path,  # dataset from original sdf2sdf paper, reference frame
            live_frame_path,  # dataset from original sdf2sdf paper, current frame
            image_pixel_row, field_size, offset, camera
        )

        # depth_interpolation_method = tsdf.DepthInterpolationMethod.NONE
        out_path = "output/test_rigid_out"
        sampling.set_focus_coordinates(0, 0)
        narrow_band_width_voxels = 2.
        iteration = 40
        optimizer = sdf2sdfo.Sdf2SdfOptimizer2d(
            verbosity_parameters=sdf2sdfo.Sdf2SdfOptimizer2d.VerbosityParameters(
                print_max_warp_update=False,
                print_iteration_energy=False
            ),
            visualization_parameters=sdf2sdfv.Sdf2SdfVisualizer.Parameters(
                out_path=out_path,
                save_initial_fields=False,
                save_final_fields=False,
                save_live_progression=True
            )
        )
        optimizer.optimize(data_to_use, narrow_band_width_voxels=narrow_band_width_voxels, iteration=iteration)
        expected_twist = np.array([[-0.079572],
                                   [0.006052],
                                   [0.159114]])
        twist = optimizer.optimize(data_to_use, narrow_band_width_voxels=narrow_band_width_voxels, iteration=iteration)

        self.assertTrue(np.allclose(expected_twist, twist, atol=10e-6))
