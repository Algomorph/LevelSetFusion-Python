//  ================================================================
//  Created by Gregory Kramida on 10/26/18.
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

#include "gradients.hpp"
#include "typedefs.hpp"

namespace math {

template <typename LaplaceOperatorFunctor>
inline
void vector_field_laplace_aux(const math::MatrixXv2f& field, math::MatrixXv2f& gradient){
	eigen_assert(false && "not implemented");
}

void vector_field_laplace(const math::MatrixXv2f& field, math::MatrixXv2f& gradient){
	eigen_assert(false && "not implemented");
}

void vector_field_negative_laplace(const math::MatrixXv2f& field, math::MatrixXv2f& gradient){
	eigen_assert(false && "not implemented");
}

/**
 * Compute numerical gradient of given matrix. Uses forward and backward differences at the boundary matrix entries,
 * central difference formula for all other entries.
 * Reference: https://en.wikipedia.org/wiki/Numerical_differentiation ;
 * Central difference formula: (f(x + h) - f(x-h))/2h, with h = 1 matrix grid cell
 *
 * @param[in] field scalar field representing the implicit function to differentiate
 * @param[out] live_gradient_x output gradient along the x axis
 * @param[out] live_gradient_y output gradient along the y axis
 */
void scalar_field_gradient(const eig::MatrixXf& field, eig::MatrixXf& live_gradient_x, eig::MatrixXf& live_gradient_y) {

	eig::Index column_count = field.cols();
	eig::Index row_count = field.rows();

	live_gradient_x = eig::MatrixXf(row_count, column_count);
	live_gradient_y = eig::MatrixXf(row_count, column_count);

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		float prev_row_val = field(0, i_col);
		float row_val = field(1, i_col);
		live_gradient_y(0, i_col) = row_val - prev_row_val;
		eig::Index i_row;
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			float next_row_val = field(i_row + 1, i_col);
			live_gradient_y(i_row, i_col) = 0.5 * (next_row_val - prev_row_val);
			prev_row_val = row_val;
			row_val = next_row_val;
		}
		live_gradient_y(i_row, i_col) = row_val - prev_row_val;
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		float prev_col_val = field(i_row, 0);
		float col_val = field(i_row, 1);
		live_gradient_x(i_row, 0) = col_val - prev_col_val;
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			float next_col_val = field(i_row, i_col + 1);
			live_gradient_x(i_row, i_col) = 0.5 * (next_col_val - prev_col_val);
			prev_col_val = col_val;
			col_val = next_col_val;
		}
		live_gradient_x(i_row, i_col) = col_val - prev_col_val;
	}
}

/**
 * Compute numerical gradient of given matrix. Uses forward and backward differences at the boundary matrix entries,
 * central difference formula for all other entries.
 * Reference: https://en.wikipedia.org/wiki/Numerical_differentiation ;
 * Central difference formula: (f(x + h) - f(x-h))/2h, with h = 1 matrix grid cell
 *
 * @param[in] field scalar field representing the implicit function to differentiate
 * @param[out] live_gradient_field output gradient field with vectors containing the x and the y gradient for each location
 */
void scalar_field_gradient(const eig::MatrixXf& field, math::MatrixXv2f& live_gradient_field) {
	eig::Index column_count = field.cols();
	eig::Index row_count = field.rows();
	live_gradient_field = math::MatrixXv2f(row_count, column_count);


#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		float prev_row_val = field(0, i_col);
		float row_val = field(1, i_col);
		live_gradient_field(0, i_col).y = row_val - prev_row_val;
		eig::Index i_row;
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			float next_row_val = field(i_row + 1, i_col);
			live_gradient_field(i_row, i_col).y = 0.5 * (next_row_val - prev_row_val);
			prev_row_val = row_val;
			row_val = next_row_val;
		}
		live_gradient_field(i_row, i_col).y = row_val - prev_row_val;
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		float prev_col_val = field(i_row, 0);
		float col_val = field(i_row, 1);
		live_gradient_field(i_row, 0).x = col_val - prev_col_val;
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			float next_col_val = field(i_row, i_col + 1);
			live_gradient_field(i_row, i_col).x = 0.5 * (next_col_val - prev_col_val);
			prev_col_val = col_val;
			col_val = next_col_val;
		}
		live_gradient_field(i_row, i_col).x = col_val - prev_col_val;
	}
}

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
void vector_field_gradient(const math::MatrixXv2f& field, math::MatrixXm2f& gradient) {

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
		gradient(i_row, 0).set_column(0, current_col_vector - prev_col_vector);
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			math::Vector2f next_col_vector = field(i_row, i_col + 1);
			gradient(i_row, i_col).set_column(0, 0.5 * (next_col_vector - prev_col_vector));
			prev_col_vector = current_col_vector;
			current_col_vector = next_col_vector;
		}
		gradient(i_row, i_col).set_column(0, current_col_vector - prev_col_vector);
	}
}

} // namespace math