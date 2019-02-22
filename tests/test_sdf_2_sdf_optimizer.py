from unittest import TestCase
import numpy as np
from calib.camera import DepthCamera
from rigid_opt import sdf_2_sdf_visualizer as sdf2sdfv, sdf_2_sdf_optimizer2d as sdf2sdfo
from rigid_opt.sdf_generation import ArrayBasedSingleFrameDataset
import utils.sampling as sampling
import math


class MyTestCase(TestCase):

    def test_sdf_2_sdf_optimizer01(self):
        canonical_frame = np.array([
             math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
             math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
             math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
             math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
             495.,              489.,              481.75,            476.75,            472.74996948,
             469.25003052,      466.5,             461.25,            458.,              455.75003052,
             451.,              449.99996948,      449.75003052,      449.50003052,      449.50003052,
             449.25,            449.25,            449.25,            449.50003052,      449.50003052,
             449.50003052,      449.99996948,      451.,              466.25,            465.5,
             464.5,             464.25003052,      463.49996948,      463.25,            462.75,
             462.50003052,      462.25,            462.,              461.75,            461.5,
             461.5,             461.5,             461.25,            461.25,            461.25,
             461.25,            461.5,             461.5,             461.75,            462.,
             462.25,            462.50003052,      462.75,            463.49996948,      464.5,
             464.25003052,      463.49996948,      463.00003052,      462.50003052,      462.25,
             461.75,            461.75,            461.5,             461.5,             461.25,
             461.25,            461.25,            461.25,            461.5,             461.5,
             461.75,            462.,              462.,              462.25,            462.50003052,
             463.00003052,      463.25,            463.75,            464.25003052,      464.75,
             465.75,            466.25,            467.,              468.25,            450.5,
             449.99996948,      449.50003052,      449.50003052,      449.50003052,      449.25,
             449.25,            449.50003052,      449.50003052,      449.50003052,      449.99996948,
             451.,              453.75,            457.50003052,      460.99996948,      464.25003052,
             467.,              472.5,             476.50003052,      478.75,            488.75003052,
             497.5,             math.inf,          math.inf,          math.inf,          math.inf,
             math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
             math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
             math.inf,          math.inf,          math.inf,          math.inf,          math.inf])

        live_frame = np.array([
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
              473.25,            458.5,             456.,              454.99996948,      454.75,
              454.5,             454.24996948,      454.00003052,      454.00003052,      454.00003052,
              453.75,            453.75,            454.5,             455.25,            459.24996948,
              464.,              467.50003052,      470.,              469.25003052,      467.74996948,
              467.,              466.5,             465.99996948,      465.5,             465.00003052,
              464.75,            464.25003052,      464.,              463.75,            463.49996948,
              463.25,            463.25,            463.00003052,      463.00003052,      463.00003052,
              463.00003052,      463.00003052,      463.25,            463.25,            463.49996948,
              463.75,            464.25003052,      464.75,            464.5,             463.75,
              463.00003052,      462.25,            461.75,            461.5,             461.25,
              460.99996948,      460.75003052,      460.5,             460.25,            460.25,
              460.25,            460.25,            460.25,            460.25,            460.25,
              460.25,            460.5,             460.5,             460.75003052,      460.99996948,
              461.25,            461.75,            462.,              462.50003052,      463.00003052,
              446.75,            446.25,            446.,              445.74996948,      445.74996948,
              445.5,             445.5,             445.5,             445.5,             445.74996948,
              446.,              448.,              449.75003052,      451.75,            453.5,
              455.5,             457.25,            459.75,            461.5,             462.50003052,
              465.25,            467.25,            469.5,             471.5,             473.,
              476.25,            478.75,            482.00003052,      483.5,             489.25,
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf,
              math.inf,          math.inf,          math.inf,          math.inf,          math.inf])

        intrinsic_matrix = np.array([[570.3999633789062, 0, 70],  # FX = 570.3999633789062 CX = 320.0
                                     [0, 570.3999633789062, 240],  # FY = 570.3999633789062 CY = 240.0
                                     [0, 0, 1]], dtype=np.float32)
        camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(480, 140), # (480, 640)
                                                               intrinsic_matrix=intrinsic_matrix))
        field_size = 32
        offset = np.array([-16, -16, 93.4375])
        data_to_use = ArrayBasedSingleFrameDataset(
            canonical_frame,  # dataset from original sdf2sdf paper, reference frame
            live_frame,  # dataset from original sdf2sdf paper, current frame
            0, field_size, offset, camera
        )

        # depth_interpolation_method = tsdf.DepthInterpolationMethod.NONE
        out_path = "output/sdf2sdf"
        sampling.set_focus_coordinates(0, 0)
        narrow_band_width_voxels = 1.
        iteration = 20
        optimizer = sdf2sdfo.Sdf2SdfOptimizer2d(
            verbosity_parameters=sdf2sdfo.Sdf2SdfOptimizer2d.VerbosityParameters(
                print_max_warp_update=False,
                print_iteration_energy=False
            ),
            visualization_parameters=sdf2sdfv.Sdf2SdfVisualizer.Parameters(
                out_path=out_path,
                save_initial_fields=False,
                save_final_fields=False,
                save_live_progression=False
            )
        )
        expected_twist = np.array([[0.0],
                                   [0.0],
                                   [0.15707963267]])
        twist = optimizer.optimize(data_to_use, narrow_band_width_voxels=narrow_band_width_voxels, iteration=iteration)

        self.assertTrue(np.allclose(expected_twist, twist, atol=1.e-1))
