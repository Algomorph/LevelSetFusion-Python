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
import interpolation as ipt
import importlib.machinery

cpp_extension = \
    importlib.machinery.ExtensionFileLoader(
        "level_set_fusion_optimization",
        "../cpp/cmake-build-release/" +
        "level_set_fusion_optimization.cpython-35m-x86_64-linux-gnu.so").load_module()


class InterpolationTest(TestCase):
    def test_interpolate_warped_live(self):
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
        expected_warp_field = np.stack((expected_u_vectors, expected_v_vectors), axis=2)
        ipt.interpolate_warped_live(canonical_field, warped_live_field, warp_field, gradient_field,
                                    band_union_only=False, known_values_only=False, substitute_original=False)
        out_u_vectors = warp_field[:, :, 0]
        out_v_vectors = warp_field[:, :, 1]
        # self.assertTrue(np.allclose(warp_field, expected_warp_field))
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))

        # re-prep data
        warped_live_field = warped_live_template.copy()

        warped_live_field, (out_u_vectors, out_v_vectors) = cpp_extension.interpolate(warped_live_field,
                                                                                          canonical_field, u_vectors,
                                                                                          v_vectors)
        self.assertTrue(np.allclose(warped_live_field, expected_new_warped_live_field))
        self.assertTrue(np.allclose(out_u_vectors, expected_u_vectors))
        self.assertTrue(np.allclose(out_v_vectors, expected_v_vectors))

        # NOTE: not testing gradient_field -- expecting it will simply be reset at each iteration in the future (maybe)
