#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 11/21/17.
#  Copyright (c) 2017 Gregory Kramida
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
import sys
import cv2
import math

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1
SQUARE_ROOT_OF_TWO = math.sqrt(2)


def main():
    image = cv2.imread("test_image1.png")
    step_size_px = 10
    vertex_row_count = image.shape[0] // step_size_px
    vertex_col_count = image.shape[1] // step_size_px
    vertex_count = vertex_row_count * vertex_col_count

    face_row_count = vertex_row_count - 1
    face_col_count = vertex_col_count - 1

    print("Grid size: ", vertex_row_count, " x ", vertex_col_count)
    print("Vertex count: ", vertex_count)
    warp_coefficient_count = 2 * vertex_count

    # G = np.zeros((2 * face_col_count * face_row_count, vertex_col_count * vertex_row_count), np.float32)
    G = np.zeros(
        (face_col_count * vertex_row_count + face_row_count * vertex_col_count, vertex_count),
        np.float32)

    ix_G_row = 0
    for ix_dx_row in range(vertex_row_count):
        for ix_dx_col in range(face_col_count):
            col_index0 = vertex_col_count * ix_dx_row + ix_dx_col
            col_index1 = col_index0 + 1
            G[ix_G_row, col_index0] = -1.0
            G[ix_G_row, col_index1] = 1.0
            ix_G_row += 1
    for ix_dy_row in range(face_row_count):
        for ix_dy_col in range(vertex_col_count):
            col_index0 = vertex_col_count * ix_dy_row + ix_dy_col
            col_index1 = col_index0 + vertex_col_count
            G[ix_G_row, col_index0] = -1.0
            G[ix_G_row, col_index1] = 1.0
            ix_G_row += 1

    P = np.vstack((np.hstack((2 * G, np.zeros_like(G))),
                   np.hstack((SQUARE_ROOT_OF_TWO * G, SQUARE_ROOT_OF_TWO * G)),
                   np.hstack((np.zeros_like(G), 2 * G))))

    constraint_count = 2
    constraint_orig_coords = np.array([[200, 100],
                                       [100, 230]], np.int32)
    constraint_final_coords = np.array([[204, 112],
                                        [106, 225]], np.int32)
    # u_0 v_0
    # u_1 v_1
    constraint_transform = constraint_final_coords - constraint_orig_coords

    constraint_coefficient_coords = (constraint_orig_coords[0, :] // step_size_px) * \
                                    vertex_col_count + (constraint_orig_coords[1, :] // step_size_px)

    I_k = np.zeros((2 * constraint_count, 2 * vertex_count), np.float32)

    # row 0: u_0 constraint
    # row 1: u_1 constraint
    # row_2: v_0 constraint
    # row 3: v_1 constraint
    I_k[0, constraint_coefficient_coords[0]] = 1.0
    I_k[1, constraint_coefficient_coords[1]] = 1.0
    I_k[2, vertex_count + constraint_coefficient_coords[0]] = 1.0
    I_k[3, vertex_count + constraint_coefficient_coords[1]] = 1.0

    U_tilde = np.rollaxis(constraint_transform, 1).reshape((-1, 1)).astype(np.float64)

    lambda_coeff = 0.9
    lambda_squared = lambda_coeff * lambda_coeff

    RHS = P.T.dot(P) + lambda_squared * I_k.T.dot(I_k)

    LHS = lambda_squared * I_k.T.dot(U_tilde)

    warp = np.linalg.solve(RHS, LHS)

    print(warp)
    print(U_tilde)
    print(warp[constraint_coefficient_coords[0]])



    return EXIT_CODE_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
