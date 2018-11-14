/*
 * statistics.cpp
 *
 *  Created on: Nov 14, 2018
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

#include "statistics.hpp"
#include "tensors.hpp"
#include "vector_operations.hpp"
#include "../traversal/field_traversal_cpu.hpp"

namespace math{

void locate_max_norm(float& max_norm, Vector2i& coordinate, const MatrixXv2f& vector_field){
	float max_squared_norm = 0;
	coordinate = math::Vector2i(0);
	int column_count = static_cast<int>(vector_field.cols());

	for(eig::Index i_element = 0; i_element < vector_field.size(); i_element++){
		float squared_length = math::squared_sum(vector_field(i_element));
		if(squared_length > max_squared_norm){
			max_squared_norm = squared_length;
			div_t division_result = div(static_cast<int>(i_element), column_count);
			coordinate.x = division_result.quot;
			coordinate.y = division_result.rem;
		}
	}
	max_norm = std::sqrt(max_squared_norm);
}

void locate_max_norm2(float& max_norm, Vector2i& coordinate, const MatrixXv2f& vector_field){
	float max_squared_norm = 0;
	coordinate = math::Vector2i(0);
	int column_count = static_cast<int>(vector_field.cols());

	auto max_norm_functor = [&] (math::Vector2f element, eig::Index i_element){
		float squared_length = math::squared_sum(element);
		if(squared_length > max_squared_norm){
			max_squared_norm = squared_length;
			div_t division_result = div(static_cast<int>(i_element), column_count);
			coordinate.x = division_result.quot;
			coordinate.y = division_result.rem;
		}
	};
	traversal::traverse_2d_field_i_element_singlethreaded(vector_field,max_norm_functor);
	max_norm = std::sqrt(max_squared_norm);
}

float ratio_of_vector_lengths_above_threshold(const MatrixXv2f& vector_field, float threshold){
	int column_count = static_cast<int>(vector_field.cols());
	float threshold_squared = threshold*threshold;
	long count_above = 0;
	long total_count = vector_field.size();

	for(eig::Index i_element = 0; i_element < vector_field.size(); i_element++){
		float squared_length = math::squared_sum(vector_field(i_element));
		if(squared_length > threshold_squared){
			count_above++;
		}
	}
	return static_cast<double>(count_above) / static_cast<double>(total_count);
}

float mean_vector_length(const MatrixXv2f& vector_field){
	int column_count = static_cast<int>(vector_field.cols());
	long total_count = vector_field.size();
	double total_length = 0.0;

	for(eig::Index i_element = 0; i_element < vector_field.size(); i_element++){
		float length = math::length(vector_field(i_element));
		total_length += static_cast<double>(length);
	}
	return static_cast<float>(total_length / static_cast<double>(total_count));
}

void mean_and_std_vector_length(float& mean, float& standard_deviation, const MatrixXv2f& vector_field){
	int column_count = static_cast<int>(vector_field.cols());
	long total_count = vector_field.size();
	double total_length = 0.0;

	for(eig::Index i_element = 0; i_element < vector_field.size(); i_element++){
		float length = math::length(vector_field(i_element));
		total_length += static_cast<double>(length);
	}
	mean = static_cast<float>(total_length / static_cast<double>(total_count));
	double total_squared_deviation = 0.0;
	for(eig::Index i_element = 0; i_element < vector_field.size(); i_element++){
		float length = math::length(vector_field(i_element));
		float local_deviation = length - mean;
		total_squared_deviation += local_deviation*local_deviation;
	}
	standard_deviation = static_cast<float>(sqrt(total_squared_deviation/static_cast<double>(total_count)));
}

}//namespace math


