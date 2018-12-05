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
#include "../math/vector_operations.hpp"
#include "../math/typedefs.hpp"
#include "boolean_operations.hpp"


namespace nonrigid_optimization {


template<bool TSkipTruncated>
void compute_tikhonov_regularization_gradient_aux(math::MatrixXv2f& gradient, float& energy,
                                                  const math::MatrixXv2f& warp_field,
                                                  const eig::MatrixXf* live_field,
                                                  const eig::MatrixXf* canonical_field) {
	eig::Index column_count = warp_field.cols();
	eig::Index row_count = warp_field.rows();
	gradient = math::MatrixXv2f(row_count, column_count);
	energy = 0.0f;

	math::Vector2<float> some_vector(1.0f,3.0f);

	auto is_truncated = [&](eig::Index i_row, eig::Index i_col) {
		return TSkipTruncated && is_outside_narrow_band((*live_field)(i_row, i_col), (*canonical_field)(i_row, i_col));
	};

#pragma omp parallel for reduction(+:energy)
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		math::Vector2f prev_row_vector = warp_field(0, i_col);
		math::Vector2f row_vector = warp_field(1, i_col);

		if(is_truncated(0, i_col)){
			gradient(0, i_col) =  math::Vector2f(0.0f);
		}else{
			math::Vector2f local_gradient_y = row_vector - prev_row_vector; //row-wise local gradient
			energy += math::squared_sum(local_gradient_y);
			//same as replicating the prev_row_vector to the border and doing -(row_vector - 2*prev_row_vector + prev_row_vector)
			gradient(0, i_col) =  -row_vector + prev_row_vector;
		}

		eig::Index i_row;
		for (i_row = 1; i_row < row_count - 1; i_row++) {
			math::Vector2f next_row_vector = warp_field(i_row + 1, i_col);
			if(is_truncated(i_row, i_col)){
				gradient(i_row, i_col) = math::Vector2f(0.0f);
			}else{
				math::Vector2f local_gradient_y = 0.5 * (next_row_vector - prev_row_vector);  //row-wise local gradient
				energy += math::squared_sum(local_gradient_y);
				//previous/next column values will be used in next loop
				gradient(i_row, i_col) = -next_row_vector + 2 * row_vector - prev_row_vector;
			}
			prev_row_vector = row_vector;
			row_vector = next_row_vector;
		}

		if(is_truncated(i_row,i_col)){
			gradient(i_row, i_col) = math::Vector2f(0.0f);
		}else{
			math::Vector2f local_gradient_y = row_vector - prev_row_vector; //row-wise local gradient
			energy += math::squared_sum(local_gradient_y);
			//same as replicating the row_vector to the border and doing -(row_vector - 2*row_vector + prev_row_vector)
			gradient(i_row, i_col) = -prev_row_vector + row_vector;
		}
	}
#pragma omp parallel for reduction(+:energy)
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		math::Vector2f prev_col_vector = warp_field(i_row, 0);
		math::Vector2f col_vector = warp_field(i_row, 1);

		if(!is_truncated(i_row, 0)){
			math::Vector2f local_gradient_x = col_vector - prev_col_vector; //column-wise local gradient
			energy += math::squared_sum(local_gradient_x);
			//same as replicating the prev_col_vector to the border and doing -(col_vector - 2*prev_col_vector + prev_col_vector)
			gradient(i_row, 0) += -col_vector + prev_col_vector;
		}
		eig::Index i_col;
		for (i_col = 1; i_col < column_count - 1; i_col++) {
			math::Vector2f next_col_vector = warp_field(i_row, i_col + 1);
			if(!is_truncated(i_row, i_col)){
				math::Vector2f local_gradient_x = 0.5 * (next_col_vector - prev_col_vector); //column-wise local gradient
				energy += math::squared_sum(local_gradient_x);
				gradient(i_row, i_col) += -next_col_vector + 2 * col_vector - prev_col_vector;
			}
			prev_col_vector = col_vector;
			col_vector = next_col_vector;
		}
		if(!is_truncated(i_row,i_col)) {
			math::Vector2f local_gradient_x = col_vector - prev_col_vector; //column-wise local gradient
			energy += math::squared_sum(local_gradient_x);
			//same as replicating the col_vector to the border and doing -(prev_col_vector - 2*col_vector + col_vector)
			gradient(i_row, i_col) += -prev_col_vector + col_vector;
		}
	}
	energy *= 0.5f;
}

void
compute_tikhonov_regularization_gradient(math::MatrixXv2f& gradient, float& energy,
                                         const math::MatrixXv2f& warp_field) {
	return compute_tikhonov_regularization_gradient_aux<false>(gradient, energy, warp_field, nullptr, nullptr);
}

void
compute_tikhonov_regularization_gradient_within_band_union(math::MatrixXv2f& gradient, float& energy,
                                                           const math::MatrixXv2f& warp_field,
                                                           const eig::MatrixXf& live_field,
                                                           const eig::MatrixXf& canonical_field) {
	return compute_tikhonov_regularization_gradient_aux<true>(gradient, energy, warp_field, &live_field,
	                                                           &canonical_field);
}

}//nonrigid_optimization