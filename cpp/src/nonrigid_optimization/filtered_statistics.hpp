/*
 * filtered_statistics.hpp
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
//libraries
#include <Eigen/Eigen>

//local
#include "../math/tensors.hpp"
#include "../math/typedefs.hpp"

#pragma once

namespace eig = Eigen;

namespace nonrigid_optimization {
float ratio_of_vector_lengths_above_threshold_band_union(const math::MatrixXv2f& vector_field, float threshold,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field);
void mean_and_std_vector_length_band_union(float& mean, float& standard_deviation,
		const math::MatrixXv2f& vector_field, const eig::MatrixXf& warped_live_field,
		const eig::MatrixXf& canonical_field);

} //namespace nonrigid_optimization
