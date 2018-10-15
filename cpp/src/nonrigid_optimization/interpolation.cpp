//  ================================================================
//  Created by Gregory Kramida on 10/11/18.
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
#include "interpolation.hpp"
#include <cstdlib>

namespace interpolation {

inline float sample_tsdf_value_at(const eig::MatrixXf& tsdf_field, int x, int y) {
	if (x < 0 || x >= tsdf_field.cols() || y < 0 || y >= tsdf_field.rows()) {
		return 1.0;
	}
	return tsdf_field(x, y);
}

inline float sample_tsdf_value_replacing_when_out_of_bounds(const eig::MatrixXf& tsdf_field, int x, int y,
                                                            float replacement_value) {
	if (x < 0 || x >= tsdf_field.cols() || y < 0 || y >= tsdf_field.rows()) {
		return replacement_value;
	}
	return tsdf_field(x, y);
}

eig::MatrixXf interpolate(const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
                          eig::MatrixXf& warp_field_u, eig::MatrixXf& warp_field_v,
                          bool band_union_only, bool known_values_only,
                          bool substitute_original, float truncation_float_threshold) {
	int matrix_size = static_cast<int>(warped_live_field.size());
	int row_length = static_cast<int>(warped_live_field.cols());
	int rows = static_cast<int>(warped_live_field.rows());

	eig::MatrixXf new_live_field(rows,row_length);

	#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		// Matrices in Eigen are column-major by default, but the converter knows to transfer to row-major
		// i_element = y * row_length + x
		div_t division_result = div(i_element, row_length);
		int x = division_result.rem;
		int y = division_result.quot;
		float live_tsdf_value = warped_live_field[i_element];
		if (band_union_only) {
			float canonical_tsdf_value = canonical_field[i_element];
			if (std::abs(live_tsdf_value) == 1.0 or std::abs(canonical_tsdf_value) == 1.0) {
				continue;
			}
		}
		if (known_values_only) {
			//TODO assumes 1.0 is the default value
			if (std::abs(live_tsdf_value) == 1.0) {
				continue;
			}
		}
		float lookup_x = x + warp_field_u(i_element);
		float lookup_y = y + warp_field_v(i_element);
		int base_x = static_cast<int>(lookup_x);
		int base_y = static_cast<int>(lookup_y);
		float ratio_x = lookup_x - base_x;
		float ratio_y = lookup_y - base_y;
		float inverse_ratio_x = 1.0F - ratio_x;
		float inverse_ratio_y = 1.0F - ratio_y;

		float value00,value01,value10,value11;
		if (substitute_original) {
			value00 = sample_tsdf_value_replacing_when_out_of_bounds(warped_live_field, base_x, base_y,
			                                                               live_tsdf_value);
			value01 = sample_tsdf_value_replacing_when_out_of_bounds(warped_live_field, base_x, base_y + 1,
			                                                               live_tsdf_value);
			value10 = sample_tsdf_value_replacing_when_out_of_bounds(warped_live_field, base_x + 1, base_y,
			                                                               live_tsdf_value);
			value11 = sample_tsdf_value_replacing_when_out_of_bounds(warped_live_field, base_x + 1, base_y + 1,
			                                                               live_tsdf_value);
		} else {
			value00 = sample_tsdf_value_at(warped_live_field, base_x, base_y);
			value01 = sample_tsdf_value_at(warped_live_field, base_x, base_y + 1);
			value10 = sample_tsdf_value_at(warped_live_field, base_x + 1, base_y);
			value11 = sample_tsdf_value_at(warped_live_field, base_x + 1, base_y + 1);
		}

		float interpolated_value0 = value00 * inverse_ratio_y + value01 * ratio_y;
		float interpolated_value1 = value10 * inverse_ratio_y + value11 * ratio_y;
		float new_value = interpolated_value0 * inverse_ratio_x + interpolated_value1 * inverse_ratio_x;
		if (std::abs(new_value) - 1.0 < truncation_float_threshold){
			new_value = std::copysign(1.0f, new_value);
			warp_field_u(i_element) = 0.0f; // probably won't work in python due to conversions, test
			warp_field_v(i_element) = 0.0f;

		}


	}
}
}