#  ================================================================
#  Created by Gregory Kramida on 10/18/18.
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
# stlib
from unittest import TestCase
# libraries
import numpy as np

# test targets
from nonrigid_opt.slavcheva.data_term import DataTermMethod
from nonrigid_opt.slavcheva.slavcheva_optimizer2d import SlavchevaOptimizer2d, ComputeMethod, AdaptiveLearningRateMethod
import utils.sampling as sampling
from nonrigid_opt.slavcheva.slavcheva_visualizer import SlavchevaVisualizer
from nonrigid_opt.slavcheva.smoothing_term import SmoothingTermMethod
from nonrigid_opt.slavcheva.sobolev_filter import generate_1d_sobolev_kernel
import level_set_fusion_optimization as cpp_module


def make_optimizer(compute_method, field_size, max_iterations=1):
    view_scaling_factor = 1024 // field_size
    optimizer = SlavchevaOptimizer2d(out_path="output/test_non_rigid_out",
                                     field_size=field_size,
                                     default_value=1,

                                     compute_method=compute_method,

                                     level_set_term_enabled=False,
                                     sobolev_smoothing_enabled=True,

                                     data_term_method=DataTermMethod.BASIC,
                                     smoothing_term_method=SmoothingTermMethod.TIKHONOV,
                                     adaptive_learning_rate_method=AdaptiveLearningRateMethod.NONE,

                                     data_term_weight=1.0,
                                     smoothing_term_weight=0.2,
                                     isomorphic_enforcement_factor=0.1,
                                     level_set_term_weight=0.2,

                                     maximum_warp_length_lower_threshold=0.05,
                                     max_iterations=max_iterations,

                                     sobolev_kernel=generate_1d_sobolev_kernel(size=3, strength=0.1),
                                     visualization_settings=SlavchevaVisualizer.Settings(
                                         enable_component_fields=True,
                                         view_scaling_factor=view_scaling_factor),
                                     enable_convergence_status_logging=True
                                     )
    return optimizer


class TestNonRigidOptimization(TestCase):
    def test_nonrigid_optimization01(self):
        # corresponds to test case test_sobolev_optimizer01 for C++
        sampling.set_focus_coordinates(0, 0)
        field_size = 4
        live_field_template = np.array([[1., 1., 0.49999955, 0.42499956],
                                        [1., 0.44999936, 0.34999937, 0.32499936],
                                        [1., 0.35000065, 0.25000066, 0.22500065],
                                        [1., 0.20000044, 0.15000044, 0.07500044]], dtype=np.float32)
        live_field = live_field_template.copy()
        canonical_field = np.array([[1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                                    [1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                                    [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                                    [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32)

        expected_live_field_out = np.array([[1., 1., 0.49408937, 0.4321034],
                                            [1., 0.44113636, 0.34710377, 0.32715625],
                                            [1., 0.3388706, 0.24753733, 0.22598255],
                                            [1., 0.21407352, 0.16514614, 0.11396749]], dtype=np.float32)
        expected_warps_out = np.array([[[0., 0.],
                                        [0., 0.],
                                        [0.03714075, 0.02109198],
                                        [0.01575381, 0.01985904]],

                                       [[0., 0.],
                                        [0.0454952, 0.04313552],
                                        [0.01572882, 0.02502392],
                                        [0.00634488, 0.02139519]],

                                       [[0., 0.],
                                        [0.07203466, 0.02682102],
                                        [0.01575179, 0.0205336],
                                        [0.00622413, 0.02577237]],

                                       [[0., 0.],
                                        [0.05771814, 0.02112256],
                                        [0.01468342, 0.01908935],
                                        [0.01397111, 0.02855439]]], dtype=np.float32)
        live_field = live_field_template.copy()
        optimizer = make_optimizer(ComputeMethod.VECTORIZED, field_size, 1)
        optimizer.optimize(live_field, canonical_field)
        self.assertTrue(np.allclose(live_field, expected_live_field_out))
        optimizer = make_optimizer(ComputeMethod.DIRECT, field_size, 1)
        live_field = live_field_template.copy()
        optimizer.optimize(live_field, canonical_field)
        self.assertTrue(np.allclose(live_field, expected_live_field_out))

    def test_nonrigid_optimization02(self):
        sampling.set_focus_coordinates(0, 0)
        field_size = 4
        live_field_template = np.array([[1., 1., 0.49999955, 0.42499956],
                                        [1., 0.44999936, 0.34999937, 0.32499936],
                                        [1., 0.35000065, 0.25000066, 0.22500065],
                                        [1., 0.20000044, 0.15000044, 0.07500044]], dtype=np.float32)
        live_field = live_field_template.copy()
        canonical_field = np.array([[1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                                    [1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                                    [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                                    [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32)
        optimizer = make_optimizer(ComputeMethod.DIRECT, field_size, 2)
        optimizer.optimize(live_field, canonical_field)
        expected_live_field_out = np.array(
            [[1., 1., 0.48917317, 0.43777004],
             [1., 0.43342987, 0.3444094, 0.3287867],
             [1., 0.33020678, 0.24566807, 0.22797936],
             [1., 0.2261582, 0.17907946, 0.14683424]], dtype=np.float32)

        report1 = optimizer.get_convergence_report()

        self.assertTrue(np.allclose(live_field, expected_live_field_out))
        live_field = live_field_template.copy()
        optimizer = make_optimizer(ComputeMethod.VECTORIZED, field_size, 2)
        optimizer.optimize(live_field, canonical_field)
        self.assertTrue(np.allclose(live_field, expected_live_field_out))

        report2 = optimizer.get_convergence_report()
        self.assertTrue(report1 == report2)

        expected_warp_stats = \
            cpp_module.WarpDeltaStatistics(0.272727, 0.0, 0.0684823, 0.0364445,
                                           0.0167321, cpp_module.Vector2i(1, 2), False, False)
        expected_diff_stats = \
            cpp_module.TsdfDifferenceStatistics(0, 0.246834, 0.111843, 0.0838871, cpp_module.Vector2i(3, 3))

        expected_report = cpp_module.ConvergenceReport(2, True, expected_warp_stats, expected_diff_stats)

        self.assertTrue(report2 == expected_report)
