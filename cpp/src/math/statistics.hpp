/*
 * warp_statistics.hpp
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

#pragma once

#include "tensors.hpp"
#include "typedefs.hpp"

namespace math{
	void locate_max_norm(float& max_norm, Vector2i& coordinate, const MatrixXv2f& vector_field);
	void locate_max_norm2(float& max_norm, Vector2i& coordinate, const MatrixXv2f& vector_field);
	float ratio_of_vector_lengths_above_threshold(const MatrixXv2f& vector_field, float threshold);
	float mean_vector_length(const MatrixXv2f& vector_field);
	void mean_and_std_vector_length(float& mean, float& standard_deviation, const MatrixXv2f& vector_field);
}

