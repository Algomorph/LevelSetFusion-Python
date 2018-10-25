//  ================================================================
//  Created by Gregory Kramida on 10/23/18.
//  Copyright (c) 2018 Gregory Kramida
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//  ================================================================

//local
#include "smoothing_term.hpp"
#include "../math/math.hpp"


namespace smoothing_term {

/**
 * \brief Computes gradient of the given 2D vector field
 * \details If the input vector field contains vectors of the form [u v]*, the output
 * will contain, for each corresponding location, the Jacobian 2x2 matrix
 *       |u_x u_y|
 *       |v_x v_y|
 * Gradients are computed numerically using central differences, with forward & backward differences applied to the
 * boundary values.
 * \param field a matrix of 2-entry vectors
 * \param gradient numerical gradient of the input matrix
 */
void gradient(const math::MatrixXv2f& field, math::MatrixXm2f& gradient) {

	eig::Index row_count = field.rows();
	eig::Index column_count = field.cols();

	gradient = math::MatrixXm2f(row_count,column_count);

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		math::Vector2f prev_row_vector = field(0, i_col);
		math::Vector2f current_row_vector = field(1, i_col);
		gradient(0, i_col).set_column(1, current_row_vector - prev_row_vector);
		eig::Index i_row;
		//traverse each column in vertical (y) direction
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			math::Vector2f next_row_vector = field(i_row + 1, i_col);
			gradient(i_row, i_col).set_column(1, 0.5 * (next_row_vector - prev_row_vector));
			prev_row_vector = current_row_vector;
			current_row_vector = next_row_vector;
		}
		gradient(i_row, i_col).set_column(1, current_row_vector - prev_row_vector);
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		math::Vector2f prev_col_vector = field(i_row, 0);
		math::Vector2f current_col_vector = field(i_row, 1);
		gradient(i_row, 0).set_column(1, current_col_vector - prev_col_vector);
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			math::Vector2f next_col_vector = field(i_row, i_col + 1);
			gradient(i_row, i_col).set_column(1, 0.5 * (next_col_vector - prev_col_vector));
			prev_col_vector = current_col_vector;
			current_col_vector = next_col_vector;
		}
		gradient(i_row, i_col).set_column(1, current_col_vector - prev_col_vector);
	}
}

}//smoothing_term