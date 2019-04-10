#  ================================================================
#  Created by Gregory Kramida on 4/8/19.
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

import numpy as np


def unpad(field, unpad_width):
    if len(field.shape) == 3:
        # @formatter:off
        return field[unpad_width: field.shape[0] - unpad_width, unpad_width: field.shape[1] - unpad_width,
                     unpad_width: field.shape[2] - unpad_width].copy()  # @formatter:on
    elif len(field.shape) == 2:
        return field[unpad_width: field.shape[0] - unpad_width, unpad_width: field.shape[1] - unpad_width]


def upsample2x_linear(field):
    if len(field.shape) == 3:
        padded = np.pad(field, 1, mode='edge')

        new_field = \
            np.zeros((field.shape[0] * 2 + 2, field.shape[1] * 2 + 2, field.shape[2] * 2 + 2))

        for z_source in range(padded.shape[0] - 1):
            z_target = z_source * 2
            for y_source in range(padded.shape[1] - 1):
                y_target = y_source * 2
                for x_source in range(padded.shape[2] - 1):
                    x_target = x_source * 2
                    v000 = padded[z_source, y_source, x_source]
                    v100 = padded[z_source + 1, y_source, x_source]
                    v010 = padded[z_source, y_source + 1, x_source]
                    v110 = padded[z_source + 1, y_source + 1, x_source]
                    v001 = padded[z_source, y_source, x_source + 1]
                    v101 = padded[z_source + 1, y_source, x_source + 1]
                    v011 = padded[z_source, y_source + 1, x_source + 1]
                    v111 = padded[z_source + 1, y_source + 1, x_source + 1]

                    zv000 = 0.75 * v000 + 0.25 * v100
                    zv100 = 0.25 * v000 + 0.75 * v100
                    zv010 = 0.75 * v010 + 0.25 * v110
                    zv110 = 0.25 * v010 + 0.75 * v110
                    zv001 = 0.75 * v001 + 0.25 * v101
                    zv101 = 0.25 * v001 + 0.75 * v101
                    zv011 = 0.75 * v011 + 0.25 * v111
                    zv111 = 0.25 * v011 + 0.75 * v111

                    yv000 = 0.75 * zv000 + 0.25 * zv010
                    yv010 = 0.25 * zv000 + 0.75 * zv010
                    yv100 = 0.75 * zv100 + 0.25 * zv110
                    yv110 = 0.25 * zv100 + 0.75 * zv110
                    yv001 = 0.75 * zv001 + 0.25 * zv011
                    yv011 = 0.25 * zv001 + 0.75 * zv011
                    yv101 = 0.75 * zv101 + 0.25 * zv111
                    yv111 = 0.25 * zv101 + 0.75 * zv111

                    new_field[z_target, y_target, x_target] = 0.75 * yv000 + 0.25 * yv001
                    new_field[z_target, y_target, x_target + 1] = 0.25 * yv000 + 0.75 * yv001
                    new_field[z_target, y_target + 1, x_target] = 0.75 * yv010 + 0.25 * yv011
                    new_field[z_target, y_target + 1, x_target + 1] = 0.25 * yv010 + 0.75 * yv011
                    new_field[z_target + 1, y_target, x_target] = 0.75 * yv100 + 0.25 * yv101
                    new_field[z_target + 1, y_target, x_target + 1] = 0.25 * yv100 + 0.75 * yv101
                    new_field[z_target + 1, y_target + 1, x_target] = 0.75 * yv110 + 0.25 * yv111
                    new_field[z_target + 1, y_target + 1, x_target + 1] = 0.25 * yv110 + 0.75 * yv111

        return unpad(new_field, 1)
    else:
        raise (NotImplementedError("Cases other than 3D not yet implemented"))


def downsample2x_linear(field):
    if len(field.shape) == 3:
        if field.shape[0] % 2 != 0 or field.shape[1] % 2 != 0 or field.shape[2] % 2 != 0:
            raise ValueError("Each field dimension must be evenly divisible by 2.")

        new_field = \
            np.zeros((field.shape[0] // 2, field.shape[1] // 2, field.shape[2] // 2))
        kernel = \
            np.array([[[0.00195312, 0.00585938, 0.00585938, 0.00195312],
                       [0.00585938, 0.01757812, 0.01757812, 0.00585938],
                       [0.00585938, 0.01757812, 0.01757812, 0.00585938],
                       [0.00195312, 0.00585938, 0.00585938, 0.00195312]],

                      [[0.00585938, 0.01757812, 0.01757812, 0.00585938],
                       [0.01757812, 0.05273438, 0.05273438, 0.01757812],
                       [0.01757812, 0.05273438, 0.05273438, 0.01757812],
                       [0.00585938, 0.01757812, 0.01757812, 0.00585938]],

                      [[0.00585938, 0.01757812, 0.01757812, 0.00585938],
                       [0.01757812, 0.05273438, 0.05273438, 0.01757812],
                       [0.01757812, 0.05273438, 0.05273438, 0.01757812],
                       [0.00585938, 0.01757812, 0.01757812, 0.00585938]],

                      [[0.00195312, 0.00585938, 0.00585938, 0.00195312],
                       [0.00585938, 0.01757812, 0.01757812, 0.00585938],
                       [0.00585938, 0.01757812, 0.01757812, 0.00585938],
                       [0.00195312, 0.00585938, 0.00585938, 0.00195312]]])

        padded = np.pad(field, 1, mode='edge')

        for z_target in range(new_field.shape[0]):
            z_source = z_target * 2
            for y_target in range(new_field.shape[1]):
                y_source = y_target * 2
                for x_target in range(new_field.shape[2]):
                    x_source = x_target * 2
                    val = np.multiply(kernel,
                                  padded[z_source:z_source + 4, y_source:y_source + 4,
                                  x_source:x_source + 4]).sum()
                    new_field[z_target, y_target, x_target] = val
        return new_field
    else:
        raise (NotImplementedError("Cases other than 3D not yet implemented"))

