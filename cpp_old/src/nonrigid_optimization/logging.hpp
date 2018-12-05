/*
 *  logging.h
 *
 *  Created on: Nov 13, 2018
 *      Author: Gregory Kramida
 *  Copyright: 2018 Gregory Kramida
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

#include <Eigen/Eigen>
#include "../math/typedefs.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {
/**
 * A structure for logging characteristics of convergence (warp-threshold-based optimization) after a full optimization run
 */
struct ConvergenceStatus {
	int iteration_count = 0;
	float max_warp_length = 0.0f;
	math::Vector2i max_warp_location = math::Vector2i(0);
	bool iteration_limit_reached = false;
	bool largest_warp_below_minimum_threshold = false;
	bool largest_warp_above_maximum_threshold = false;
	ConvergenceStatus() = default;
	ConvergenceStatus(int iteration_count, float max_warp_length, math::Vector2i max_warp_location,
			bool iteration_limit_reached,
			bool largest_vector_below_minimum_threshold, bool largest_vector_above_maximum_threshold);

};

/**
 * A structure for logging statistics pertaining to warps at the end of each iteration
 * (in warp-threshold-based optimizations)
 */
struct IterationWarpStatistics {
	/* TODO: reorder -- max warp length & loc should be nearby. Have to then change constructor, python module,
	 and to_array function as well to reflect the new ordering. */
	float ratio_of_warps_above_minimum_threshold = 0.0;
	float max_warp_length = 0.0f;
	float mean_warp_length = 0.0;
	float standard_deviation_of_warp_length = 0.0;
	math::Vector2i max_warp_location = math::Vector2i(0);
	IterationWarpStatistics() = default;
	IterationWarpStatistics(float ratio_of_warps_above_minimum_threshold, float max_warp_length,
			float average_warp_length, float standard_deviation_of_warp_length, math::Vector2i max_warp_location);
	eig::VectorXf to_array();
};
} //namespace nonrigid_optimization
