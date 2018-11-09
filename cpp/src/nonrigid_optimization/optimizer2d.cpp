//  ================================================================
//  Created by Gregory Kramida on 11/2/18.
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
#include "optimizer2d.hpp"

namespace nonrigid_optimization {
void Optimizer2d::SharedParameters::set_from_json(pt::ptree root) {
	this->gradient_descent_rate = root.get<float>("gradient_descent_rate", 0.1f);

	this->maximum_warp_length_lower_threshold = root.get<float>("maximum_warp_length_lower_threshold", 0.1);
	this->maximum_warp_length_upper_threshold = root.get<float>("maximum_warp_length_upper_threshold", 10000);
	this->maximum_iteration_count = root.get<int>("maximum_iteration_count", 100);
	this->minimum_iteration_count = root.get<int>("minimum_iteration_count", 1);

	this->enable_energy_and_min_vector_logging = root.get<bool>("enable_energy_and_min_vector_logging", false);
	this->enable_focus_spot_analytics = root.get<bool>("enable_focus_spot_analytics", false);
	this->enable_live_sdf_progression_logging = root.get<bool>("enable_live_sdf_progression_logging", false);
	this->enable_gradient_logging = root.get<bool>("enable_gradient_logging", false);
	this->enable_gradient_component_logging = root.get<bool>("enable_gradient_component_logging", false);

	pt::ptree& focus_spot_root = root.get_child("focus_spot");
	this->focus_spot = math::Vector2i(focus_spot_root.get<int>("x", 0), focus_spot_root.get<int>("y", 0));
}

void Optimizer2d::SharedParameters::set_from_values(float gradient_descent_rate,
													// termination condition parameters
		                                            float maximum_warp_length_lower_threshold,
		                                            float maximum_warp_length_upper_threshold,
		                                            int maximum_iteration_count,
		                                            int minimum_iteration_count,
													// logging
		                                            bool enable_focus_spot_analytics,
		                                            bool enable_energy_and_min_vector_logging,
		                                            bool enable_live_sdf_progression_logging,
		                                            bool enable_gradient_logging,
		                                            bool enable_gradient_component_logging,
		                                            int focus_spot_x, int focus_spot_y) {
	this->gradient_descent_rate = gradient_descent_rate;

	this->maximum_warp_length_lower_threshold = maximum_warp_length_lower_threshold;
	this->maximum_warp_length_upper_threshold = maximum_warp_length_upper_threshold;
	this->maximum_iteration_count = maximum_iteration_count;
	this->minimum_iteration_count = minimum_iteration_count;

	this->enable_energy_and_min_vector_logging = enable_energy_and_min_vector_logging;
	this->enable_focus_spot_analytics = enable_focus_spot_analytics;
	this->enable_live_sdf_progression_logging = enable_live_sdf_progression_logging;
	this->enable_gradient_logging = enable_gradient_logging;
	this->enable_gradient_component_logging = enable_gradient_component_logging;

	this->focus_spot = math::Vector2i(focus_spot_x,focus_spot_y);

}


Optimizer2d::SharedParameters& Optimizer2d::shared_parameters() {
	return SharedParameters::get_instance();
}

bool Optimizer2d::are_termination_conditions_reached(int completed_iteration_count, float largest_warp_vector) {
	static SharedParameters& shared_parameters = Optimizer2d::shared_parameters();
	return completed_iteration_count >= shared_parameters.minimum_iteration_count &&
	       (completed_iteration_count >= shared_parameters.maximum_iteration_count ||
	        largest_warp_vector < shared_parameters.maximum_warp_length_lower_threshold ||
	        largest_warp_vector > shared_parameters.maximum_warp_length_upper_threshold);
}

}//namespace nonrigid_optimization
