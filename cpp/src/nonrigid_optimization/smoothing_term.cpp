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


namespace smoothing_term {

//void gradient(const eig::MatrixXf& warp_field_u, const eig::MatrixXf& warp_field_v,
//              eig::MatrixXf& live_gradient_ux, eig::MatrixXf& live_gradient_uy,
//              eig::MatrixXf& live_gradient_vx, eig::MatrixXf& live_gradient_vy) {

void gradient(const eig::Tensor<float, 3>& warp_field,
              eig::Tensor<float, 3>& warp_field_gradient) {

	eigen_assert((warp_field.dimension(2) == 2) &&
	"Tensor passed in for the warp field doesn't have a depth dimension of 2.");

	eig::Index row_count = warp_field.dimension(0);
	eig::Index column_count = warp_field.dimension(1);
	const eig::Index output_depth = 4; // u_x, u_y, v_x, v_y, where u_x means du/dx
	warp_field_gradient = eig::Tensor<float,3>(row_count,column_count,output_depth);

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		float prev_row_u = warp_field(0, i_col, 0);
		float prev_row_v = warp_field(0, i_col, 1);
		float row_u = warp_field(1, i_col, 0);
		float row_v = warp_field(1, i_col, 1);
		warp_field_gradient(0, i_col, 1) = row_u - prev_row_u; //u_y
		warp_field_gradient(0, i_col, 3) = row_u - prev_row_u; //v_y
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

}//smoothing_term