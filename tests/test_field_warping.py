#  ================================================================
#  Created by Gregory Kramida on 10/15/18.
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
from unittest import TestCase
import numpy as np
from nonrigid_opt import field_warping as ipt
import tests.test_data.hierarchical_optimizer_test_data as fixtures

import level_set_fusion_optimization as cpp


class InterpolationTest(TestCase):
    def test_warp_field_advanced01(self):
        u_vectors = np.array([[0.5, -0.5],
                              [0.5, -0.5]], dtype=np.float32)
        v_vectors = np.array([[0.5, 0.5],
                              [-0.5, -0.5]], dtype=np.float32)

        warp_field_template = np.stack((u_vectors, v_vectors), axis=2)
        warp_field = warp_field_template.copy()
        gradient_field_template = warp_field_template * 10
        gradient_field = gradient_field_template.copy()
        warped_live_template = np.array([[1., -1.],
                                         [1., -1.]], dtype=np.float32)

        warped_live_field = warped_live_template.copy()
        canonical_field = np.array([[0., 0.],
                                    [0., 0.]], dtype=np.float32)
        expected_new_warped_live_field = np.array([[0., 0.],
                                                   [0., 0.]], dtype=np.float32)
        expected_u_vectors = np.array([[0.5, -0.5],
                                       [0.5, -0.5]], dtype=np.float32)
        expected_v_vectors = np.array([[0.5, 0.5],
                                       [-0.5, -0.5]], dtype=np.float32)
        # expected_warp_field = np.stack((expected_u_vectors, expected_v_vectors), axis=2)
        warped_live_field = ipt.warp_field_advanced(canonical_field, warped_live_field, warp_field, gradient_field,
                                                    band_union_only=False, known_values_only=False,
                                                    substitute_original=False)
        out_u_vectors = warp_field[:, :, 0]
        out_v_vectors = warp_field[:, :, 1]
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))

        # re-prep data
        warped_live_field = warped_live_template.copy()

        warped_live_field, (out_u_vectors, out_v_vectors) = cpp.warp_field_advanced(warped_live_field,
                                                                           canonical_field, u_vectors,
                                                                           v_vectors)

        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))

    def test_warp_field_advanced02(self):
        # corresponds to resampling_test02 for C++
        u_vectors = np.array([[0.0, 0.0, 0.0],
                              [-.5, 0.0, 0.0],
                              [1.5, 0.5, 0.0]], dtype=np.float32)

        v_vectors = np.array([[-1.0, 0.0, 0.0],
                              [0.0, 0.0, 0.5],
                              [-1.5, 0.5, -0.5]], dtype=np.float32)

        warp_field_template = np.stack((u_vectors, v_vectors), axis=2)
        warp_field = warp_field_template.copy()
        gradient_field_template = warp_field_template * 10
        gradient_field = gradient_field_template.copy()
        warped_live_template = np.array([[1.0, 1.0, 1.0],
                                         [0.5, 1.0, 1.0],
                                         [0.5, 0.5, -1.0]], dtype=np.float32)

        warped_live_field = warped_live_template.copy()
        canonical_field = np.array([[-1.0, 0., 0.0],
                                    [0.0, 0., 1.0],
                                    [0.0, 0., 1.0]], dtype=np.float32)
        expected_new_warped_live_field = np.array([[1.0, 1.0, 1.0],
                                                   [0.5, 1.0, 1.0],
                                                   [1.0, 0.125, -1.0]], dtype=np.float32)
        expected_u_vectors = np.array([[0.0, 0.0, 0.0],
                                       [-0.5, 0.0, 0.0],
                                       [0.0, 0.5, 0.0]], dtype=np.float32)
        expected_v_vectors = np.array([[-1.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.5],
                                       [0.0, 0.5, -0.5]], dtype=np.float32)
        warped_live_field = ipt.warp_field_advanced(canonical_field, warped_live_field, warp_field, gradient_field,
                                                    band_union_only=True, known_values_only=False,
                                                    substitute_original=True)
        out_u_vectors = warp_field[:, :, 0]
        out_v_vectors = warp_field[:, :, 1]

        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))

        # re-prep data
        warped_live_field = warped_live_template.copy()

        warped_live_field, (out_u_vectors, out_v_vectors) = \
            cpp.warp_field_advanced(warped_live_field, canonical_field, u_vectors, v_vectors,
                           band_union_only=True, known_values_only=False, substitute_original=True)

        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))

    def test_warp_field_advanced03(self):
        u_vectors = np.array([[0., 0., 0., 0.],
                              [-1.95794283, 1.59443461, -0.80321548, -0.41660499],
                              [0.99072356, 0.33884474, -1.16845247, 1.46578561],
                              [0., 0., 0., 0.]], dtype=np.float32)
        v_vectors = np.array([[0., 0., 0., 0.],
                              [-0.89691237, -0.45941665, 1.36006788, -1.05888156],
                              [-0.47305308, -1.27971876, -0.38927596, -1.83300769],
                              [0., 0., 0., 0.]], dtype=np.float32)

        warp_field_template = np.stack((u_vectors, v_vectors), axis=2)
        warp_field = warp_field_template.copy()
        gradient_field_template = warp_field_template * 10
        gradient_field = gradient_field_template.copy()
        warped_live_template = np.array([[0., 0., 0., 0.],
                                         [0.22921804, 0.04988099, -0.45673641, 0.05888156],
                                         [-0.46242488, 0., -0.44900494, 0.92592539],
                                         [0., 0., 0., 0.]], dtype=np.float32)
        warped_live_field = warped_live_template.copy()
        canonical_field = np.array([[0., 0., 0., 0.],
                                    [0.18455172, 0., 0., 0.45854051],
                                    [0.4508187, 0.8420034, 0.41780566, 0.34275345],
                                    [0.90931962, 0.20844429, 0.27169698, 0.57087366]], dtype=np.float32)
        expected_new_warped_live_field = np.array([[0., 0., 0., 0.],
                                                   [1., -0.08121467, -0.05654262, 0.05888156],
                                                   [0.02212291, -0.08771848, -0.01639593, 1.],
                                                   [0., 0., 0., 0.]], dtype=np.float32)
        expected_u_vectors = np.array([[0., 0., 0., 0.],
                                       [0., 1.59443461, -0.80321548, -0.41660499],
                                       [0.99072356, 0.33884474, -1.16845247, 0.],
                                       [0., 0., 0., 0.]], dtype=np.float32)
        expected_v_vectors = np.array([[0., 0., 0., 0.],
                                       [0., -0.45941665, 1.36006788, -1.05888156],
                                       [-0.47305308, -1.27971876, -0.38927596, 0.],
                                       [0., 0., 0., 0.]], dtype=np.float32)
        warped_live_field = ipt.warp_field_advanced(canonical_field, warped_live_field, warp_field, gradient_field,
                                                    band_union_only=False, known_values_only=False,
                                                    substitute_original=False)
        out_u_vectors = warp_field[:, :, 0]
        out_v_vectors = warp_field[:, :, 1]
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))

        # re-prep data
        warped_live_field = warped_live_template.copy()

        warped_live_field, (out_u_vectors, out_v_vectors) = cpp.warp_field_advanced(warped_live_field,
                                                                           canonical_field, u_vectors,
                                                                           v_vectors)

        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))

        # NOTE: not testing gradient_field -- expecting it will simply be reset at each iteration in the future (maybe)

    def test_warp_field_advanced04(self):
        u_vectors = np.array([[-0., -0., 0.03732542, 0.01575381],
                              [-0., 0.04549519, 0.01572882, 0.00634488],
                              [-0., 0.07203466, 0.01575179, 0.00622413],
                              [-0., 0.05771814, 0.01468342, 0.01397111]], dtype=np.float32)
        v_vectors = np.array([[-0., -0., 0.02127664, 0.01985903],
                              [-0., 0.04313552, 0.02502393, 0.02139519],
                              [-0., 0.02682102, 0.0205336, 0.02577237],
                              [-0., 0.02112256, 0.01908935, 0.02855439]], dtype=np.float32)

        warp_field_template = np.stack((u_vectors, v_vectors), axis=2)
        warp_field = warp_field_template.copy()
        gradient_field_template = warp_field_template * 10
        gradient_field = gradient_field_template.copy()
        warped_live_template = np.array([[1., 1., 0.49999955, 0.42499956],
                                         [1., 0.44999936, 0.34999937, 0.32499936],
                                         [1., 0.35000065, 0.25000066, 0.22500065],
                                         [1., 0.20000044, 0.15000044, 0.07500044]], dtype=np.float32)
        warped_live_field = warped_live_template.copy()
        canonical_field = np.array([[1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                                    [1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                                    [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                                    [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32)
        expected_new_warped_live_field = np.array(
            [[1., 1., 0.49404836, 0.4321034],
             [1., 0.44113636, 0.34710377, 0.32715625],
             [1., 0.3388706, 0.24753733, 0.22598255],
             [1., 0.21407352, 0.16514614, 0.11396749]], dtype=np.float32)

        warped_live_field = ipt.warp_field_advanced(canonical_field, warped_live_field, warp_field, gradient_field,
                                                    band_union_only=False, known_values_only=False,
                                                    substitute_original=False)
        out_u_vectors = warp_field[:, :, 0]
        out_v_vectors = warp_field[:, :, 1]
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))

        # re-prep data
        warped_live_field = warped_live_template.copy()

        warped_live_field, (out_u_vectors, out_v_vectors) = cpp.warp_field_advanced(warped_live_field,
                                                                           canonical_field, u_vectors,
                                                                           v_vectors)
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))

    def test_warp_field_advanced05(self):
        u_vectors = np.array([[-0., -0., 0.0334751, 0.01388371],
                              [-0., 0.04041886, 0.0149368, 0.00573045],
                              [-0., 0.06464156, 0.01506416, 0.00579486],
                              [-0., 0.06037777, 0.0144603, 0.01164452]], dtype=np.float32)
        v_vectors = np.array([[-0., -0., 0.019718, 0.02146172],
                              [-0., 0.03823357, 0.02406227, 0.02212186],
                              [-0., 0.02261183, 0.01864575, 0.02234527],
                              [-0., 0.01906347, 0.01756042, 0.02574961]], dtype=np.float32)

        warp_field_template = np.stack((u_vectors, v_vectors), axis=2)
        warp_field = warp_field_template.copy()
        gradient_field_template = warp_field_template * 10
        gradient_field = gradient_field_template.copy()
        warped_live_template = np.array([[1., 1., 0.49404836, 0.4321034],
                                         [1., 0.44113636, 0.34710377, 0.32715625],
                                         [1., 0.3388706, 0.24753733, 0.22598255],
                                         [1., 0.21407352, 0.16514614, 0.11396749]], dtype=np.float32)
        warped_live_field = warped_live_template.copy()
        canonical_field = np.array([[1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01],
                                    [1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01],
                                    [1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02],
                                    [1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02]], dtype=np.float32)
        expected_new_warped_live_field = np.array(
            [[1., 1., 0.48910502, 0.43776682],
             [1., 0.43342987, 0.34440944, 0.3287866],
             [1., 0.33020678, 0.24566805, 0.22797936],
             [1., 0.2261582, 0.17907946, 0.14683424]], dtype=np.float32)

        warped_live_field = ipt.warp_field_advanced(canonical_field, warped_live_field, warp_field, gradient_field,
                                                    band_union_only=False, known_values_only=False,
                                                    substitute_original=False)
        out_u_vectors = warp_field[:, :, 0]
        out_v_vectors = warp_field[:, :, 1]
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))

        # re-prep data
        warped_live_field = warped_live_template.copy()

        warped_live_field, (out_u_vectors, out_v_vectors) = \
            cpp.warp_field_advanced(warped_live_field, canonical_field, u_vectors, v_vectors)
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))

    def test_warp_field01(self):
        warp_field = fixtures.warp_field_A_16x16
        scalar_field = fixtures.field_A_16x16
        resampled_field = ipt.warp_field(scalar_field, warp_field)
        self.assertTrue(np.allclose(resampled_field, fixtures.fA_resampled_with_wfA))

    def test_warp_field_replacement01(self):
        warp_field = fixtures.warp_field_B_16x16
        scalar_field = fixtures.field_B_16x16
        resampled_field = ipt.warp_field_replacement(scalar_field, warp_field, 0.0)
        print(repr(resampled_field))
        self.assertTrue(np.allclose(resampled_field, fixtures.fB_resampled_with_wfB_replacement))
