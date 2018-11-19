/*
 * filtered_statistics.cpp
 *
 *  Created on: Nov 16, 2018
 *      Author: Gregory Kramida
 *   Copyright: 2018 Gregory Kramida
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#include "filtered_statistics.hpp"
#include "boolean_operations.hpp"
#include "../math/vector_operations.hpp"

namespace nonrigid_optimization {

static inline void check_dimensions(const math::MatrixXv2f& vector_field,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field) {
	eigen_assert( vector_field.rows() == warped_live_field.rows() &&
			vector_field.cols() == warped_live_field.cols() &&
			vector_field.rows() == canonical_field.rows() &&
			vector_field.cols() == canonical_field.cols() &&
			"Dimensions of one of the input matrices don't seem to match.");
}

float ratio_of_vector_lengths_above_threshold_band_union(const math::MatrixXv2f& vector_field, float threshold,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field) {

	check_dimensions(vector_field, warped_live_field, canonical_field);

	int column_count = static_cast<int>(vector_field.cols());
	float threshold_squared = threshold * threshold;
	long count_above = 0;
	long total_count = 0;

	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		if (is_outside_narrow_band(warped_live_field(i_element), canonical_field(i_element)))
			continue;
		float squared_length = math::squared_sum(vector_field(i_element));
		if (squared_length > threshold_squared) {
			count_above++;
		}
		total_count++;
	}
	return static_cast<double>(count_above) / static_cast<double>(total_count);
}

void mean_and_std_vector_length_band_union(float& mean, float& standard_deviation, const math::MatrixXv2f& vector_field,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field) {

	check_dimensions(vector_field, warped_live_field, canonical_field);

	int column_count = static_cast<int>(vector_field.cols());
	long total_count = 0;
	double total_length = 0.0;

	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		if (is_outside_narrow_band(warped_live_field(i_element), canonical_field(i_element)))
			continue;
		float length = math::length(vector_field(i_element));
		total_length += static_cast<double>(length);
		total_count += 1;
	}

	mean = static_cast<float>(total_length / static_cast<double>(total_count));
	double total_squared_deviation = 0.0;
	for (eig::Index i_element = 0; i_element < vector_field.size(); i_element++) {
		if (is_outside_narrow_band(warped_live_field(i_element), canonical_field(i_element)))
			continue;
		float length = math::length(vector_field(i_element));
		float local_deviation = length - mean;
		total_squared_deviation += local_deviation * local_deviation;
	}
	standard_deviation = static_cast<float>(sqrt(total_squared_deviation / static_cast<double>(total_count)));
}

} //namespace nonrigid_optimization
