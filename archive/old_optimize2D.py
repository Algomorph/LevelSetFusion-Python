#  ================================================================
#  Created by Gregory Kramida on 8/24/18.
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

def grad_x(scalar_field):
    first_derivative = np.zeros((scalar_field.shape[0], scalar_field.shape[1] - 1))
    for i_scalar in range(scalar_field.shape[1] - 1):
        i_next_scalar = i_scalar + 1
        first_derivative[:, i_scalar] = scalar_field[:, i_next_scalar] - scalar_field[:, i_scalar]
    return first_derivative


def jacobian_and_hessian(scalar_field):
    """
    :param scalar_field:
    :return:
    :rtype:(numpy.ndarray,numpy.ndarray,numpy.ndarray,numpy.ndarray,numpy.ndarray)
    """
    dx = np.diff(scalar_field, axis=1)
    dx2 = grad_x(scalar_field)
    dy = np.diff(scalar_field, axis=0)
    dxx = np.diff(dx, axis=1)
    dyy = np.diff(dy, axis=0)
    dxy = np.diff(dx, axis=0)
    return dx, dy, dxx, dxy, dyy


def run_old_simulation(verbose=False):
    scalar_field_size = 5
    initial_scalar_field = np.random.rand(scalar_field_size, scalar_field_size)
    if verbose:
        print(initial_scalar_field)
    scalar_field = initial_scalar_field.copy()
    learning_rate = 0.1

    num_iterations = 0

    max_update = np.inf
    update_threshold = 0.0000001

    dx, dy, dxx, dxy, dyy = jacobian_and_hessian(scalar_field)

    while np.abs(max_update) > update_threshold and num_iterations < 10000:
        dx, dy, dxx, dxy, dyy = jacobian_and_hessian(scalar_field)
        update = np.zeros((scalar_field.shape[0], scalar_field.shape[1]))
        # update = np.zeros_like(scalar_field)
        # update = np.zeros_like(dxx)
        # for i_update in range(1,update.shape[1]-1):
        #      update[:, i_update] = -2.0 * (dx[:, i_update] + dxx[:, i_update-1])
        # print(update)
        # fill boundary values
        border_factor = -2.0

        update[0, 1:scalar_field_size - 1] = border_factor * (dx[0, 1:] + dxx[0, :])

        update[scalar_field_size - 1, 1:scalar_field_size - 1] = \
            border_factor * (dx[scalar_field_size - 1, 1:] + dxx[scalar_field_size - 1, :])

        update[1:scalar_field_size - 1, 0] = border_factor * (dy[1:, 0] + dyy[:, 0])
        update[1:scalar_field_size - 1, scalar_field_size - 1] = \
            border_factor * (dy[1:, scalar_field_size - 1] + dyy[:, scalar_field_size - 1])

        for y in range(1, update.shape[0] - 1):
            for x in range(1, update.shape[1] - 1):
                # update[y, x] = -2.0 * (dx[y, x + 1] + dxx[y, x] + dy[y + 1, x] + dyy[y, x])
                # update[y, x] = -2.0 * (dxx[y, x] + dyy[y, x])
                # update[y, x] = -2.0 * (dx[y, x+1] + dy[y+1, x])
                # update[y, x] = -2.0 * (dx[y, x] + dy[y, x])
                # update[y, x] = -2.0 * dxy[y, x]
                update[y, x] = (-dxx[y - 1, x - 1] + 2.0 * dxy[y - 1, x - 1] + -dyy[y - 1, x - 1])
                pass

        # scalar_field[1:-1, 1:-1] -= learning_rate * update
        scalar_field -= learning_rate * update
        max_update = np.max(update)
        num_iterations += 1
        energy = np.sum(dx ** 2) + np.sum(dy ** 2)
        if verbose:
            print(max_update, energy)

    if verbose:
        print(scalar_field, num_iterations)
    return num_iterations
