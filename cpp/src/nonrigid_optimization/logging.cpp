/*
 * logging.cpp
 *
 *  Created on: Nov 13, 2018
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

#include "logging.hpp"

namespace nonrigid_optimization {
ConvergenceStatus::ConvergenceStatus(int iteration_count, float max_vector_length, bool iteration_limit_reached,
		bool largest_vector_below_minimum_threshold, bool largest_vector_above_maximum_threshold)
:
		iteration_count(iteration_count), max_warp_length(max_vector_length),
				iteration_limit_reached(iteration_limit_reached),
				largest_warp_below_minimum_threshold(largest_vector_below_minimum_threshold),
				largest_warp_above_maximum_threshold(largest_vector_above_maximum_threshold) {
}

IterationWarpStatistics::IterationWarpStatistics(float ratio_of_warps_above_minimum_threshold, float max_warp_length,
		float average_warp_length, float standard_deviation_of_warp_length)
:
		ratio_of_warps_above_minimum_threshold(ratio_of_warps_above_minimum_threshold),
				max_warp_length(max_warp_length), mean_warp_length(average_warp_length),
				standard_deviation_of_warp_length(standard_deviation_of_warp_length) {
}

eig::Vector4f IterationWarpStatistics::to_array() {
	eig::Vector4f out;
	out << this->ratio_of_warps_above_minimum_threshold, this->max_warp_length, this->mean_warp_length,
			this->standard_deviation_of_warp_length;
	return out;
}

} //

