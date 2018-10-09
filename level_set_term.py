#  ================================================================
#  Created by Gregory Kramida on 9/17/18.
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
import numpy as np
from sampling import sample_at


def level_set_term_at_location(warped_live_field, x, y, epsilon=1e-5):
    live_y_minus_one = sample_at(warped_live_field, x, y - 1)
    # live_y_minus_two = sample_at(warped_live_field, x, y - 2)
    live_x_minus_one = sample_at(warped_live_field, x - 1, y)
    # live_x_minus_two = sample_at(warped_live_field, x - 2, y)
    live_y_plus_one = sample_at(warped_live_field, x, y + 1)
    # live_y_plus_two = sample_at(warped_live_field, x, y + 2)
    live_x_plus_one = sample_at(warped_live_field, x + 1, y)
    # live_x_plus_two = sample_at(warped_live_field, x + 2, y)
    live_sdf = sample_at(warped_live_field, x, y)

    live_x_minus_one_y_minus_one = sample_at(warped_live_field, x - 1, y - 1)
    live_x_plus_one_y_minus_one = sample_at(warped_live_field, x + 1, y - 1)
    live_x_minus_one_y_plus_one = sample_at(warped_live_field, x - 1, y + 1)
    live_x_plus_one_y_plus_one = sample_at(warped_live_field, x + 1, y + 1)

    x_grad = 0.5 * (live_x_plus_one - live_x_minus_one)
    y_grad = 0.5 * (live_y_plus_one - live_y_minus_one)

    grad_xx = live_x_plus_one - 2 * live_sdf + live_x_plus_one
    grad_yy = live_y_plus_one - 2 * live_sdf + live_y_plus_one
    # grad_xx = live_x_plus_two - 2*live_sdf + live_y_plus_two
    # grad_yy = live_y_plus_two - 2*live_sdf + live_y_plus_two

    grad_xy = 0.25 * (live_x_plus_one_y_plus_one - live_x_minus_one_y_plus_one -
                      live_x_plus_one_y_minus_one + live_x_minus_one_y_minus_one)

    scale_factor = 10.0  # really should equal narrow-band half-width in voxels

    gradient = np.array([[x_grad, y_grad]]).T * scale_factor
    hessian = np.array([[grad_xx, grad_xy],
                        [grad_xy, grad_yy]]) * scale_factor

    gradient_length = np.linalg.norm(gradient)
    level_set_gradient = ((1.0 - gradient_length) / (gradient_length + epsilon) * hessian.dot(gradient)).reshape(-1)
    local_energy_contribution = 0.5 * pow((gradient_length - 1.0), 2)
    return level_set_gradient, local_energy_contribution