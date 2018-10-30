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
#include "../math/typedefs.hpp"
#include "boolean_operations.hpp"


namespace nonrigid_optimization {


template<bool TSkipTruncated>
void compute_tikhonov_regularization_gradient_aux(math::MatrixXv2f& gradient, float& energy,
                                                  const math::MatrixXv2f& warp_field,
                                                  const eig::MatrixX2f* live_field,
                                                  const eig::MatrixX2f* canonical_field) {
	eig::Index column_count = warp_field.cols();
	eig::Index row_count = warp_field.rows();
	gradient = math::MatrixXv2f(row_count, column_count);

	auto is_truncated = [&](eig::Index i_row, eig::Index i_col) {
		return TSkipTruncated && is_outside_narrow_band((*live_field)(i_row, i_col), (*canonical_field)(i_row, i_col));
	};

#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		math::Vector2f prev_row_val = warp_field(0, i_col);
		math::Vector2f row_val = warp_field(1, i_col);

		//same as replicating the prev_row_val to the border and doing (nonborder_value - 2*border_value + border_value)
		gradient(0, i_col) = is_truncated(0, i_col) ? math::Vector2f(0.0f) : -row_val + prev_row_val;

		eig::Index i_row;
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			math::Vector2f next_row_val = warp_field(i_row + 1, i_col);
			//previous/next column values will be used in next loop
			gradient(i_row, i_col) = is_truncated(i_row, i_col) ? math::Vector2f(0.0f) : -next_row_val + 2 * row_val - prev_row_val;
			prev_row_val = row_val;
			row_val = next_row_val;
		}
		//same as replicating the row_val to the border and doing (row_val - 2*row_val + prev_row_val)
		gradient(i_row, i_col) = is_truncated(i_row, i_col) ? math::Vector2f(0.0f) : -prev_row_val + row_val;
	}
#pragma omp parallel for
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		math::Vector2f prev_col_val = warp_field(i_row, 0);
		math::Vector2f col_val = warp_field(i_row, 1);

		//same as replicating the prev_row_val to the border and doing (nonborder_value - 2*border_value + border_value)
		if(!is_truncated(i_row, 0)) gradient(i_row, 0) += -col_val + prev_col_val;
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			math::Vector2f next_col_val = warp_field(i_row, i_col + 1);
			if(!is_truncated(i_row, i_col)) gradient(i_row, i_col) += -next_col_val + 2 * col_val - prev_col_val;
			prev_col_val = col_val;
			col_val = next_col_val;
		}
		//same as replicating the prev_row_val to the border and doing (nonborder_value - 2*border_value + border_value)
		if(!is_truncated(i_row,i_col)) gradient(i_row, i_col) += -prev_col_val + col_val;
	}
}

void
compute_tikhonov_regularization_gradient(math::MatrixXv2f& gradient, float& energy,
                                         const math::MatrixXv2f& warp_field) {
	return compute_tikhonov_regularization_gradient_aux<false>(gradient, energy, warp_field, nullptr, nullptr);
}

void
compute_tikhonov_regularization_gradient_within_band_union(math::MatrixXv2f& gradient, float& energy,
                                                           const math::MatrixXv2f& warp_field,
                                                           const eig::MatrixX2f& live_field,
                                                           const eig::MatrixX2f& canonical_field) {
	return compute_tikhonov_regularization_gradient_aux<true>(gradient, energy, warp_field, &live_field,
	                                                           &canonical_field);
}

}//nonrigid_optimization