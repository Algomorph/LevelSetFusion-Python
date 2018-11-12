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
#pragma once

#include <boost/property_tree/ptree.hpp>
#include "../math/typedefs.hpp"


// Short alias for this namespace
namespace pt = boost::property_tree;

namespace nonrigid_optimization {

class Optimizer2d {
public:
	class SharedParameters {
	public:
		static SharedParameters& get_instance() {
			static SharedParameters instance;
			return instance;
		}

		SharedParameters(const SharedParameters&) = delete;
		void operator=(SharedParameters const&) = delete;

		// when adding new parameters, please modify the following set functions
		void set_from_json(pt::ptree root);
		void set_from_values(float gradient_descent_rate = 0.1f,

				// termination condition parameters
				             float maximum_warp_length_lower_threshold = 0.1f,
				             float maximum_warp_length_upper_threshold = 10000,
				             int maximum_iteration_count = 100,
				             int minimum_iteration_count = 1,

				// logging
				             bool enable_focus_spot_analytics = false,
				             bool enable_energy_and_min_vector_logging = false,
				             bool enable_live_sdf_progression_logging = false,
				             bool enable_gradient_logging = false,
				             bool enable_gradient_component_logging = false,
				             int focus_spot_x = 0, int focus_spot_y = 0);

		float gradient_descent_rate = 0.1f;

		// termination condition parameters
		float maximum_warp_length_lower_threshold = 0.1f;
		float maximum_warp_length_upper_threshold = 10000;
		int maximum_iteration_count = 100;
		int minimum_iteration_count = 1;

		// logging
		bool enable_focus_spot_analytics = false;
		bool enable_energy_and_min_vector_logging = false;
		bool enable_live_sdf_progression_logging = false;
		bool enable_gradient_logging = false;
		bool enable_gradient_component_logging = false;
		math::Vector2i focus_spot = math::Vector2i(0, 0);


	private:
		SharedParameters() = default;
	};

	static SharedParameters& shared_parameters();

	Optimizer2d() = default;
	virtual ~Optimizer2d() = default;

	/**
	 * \brief Perform non-rigid alignment of the live 2D TSDF field to the canonical 2D TSDF field
	 * \param live_field a scalar TSDF field (presumably obtained from raw depth image row pixel)
	 * \param canonical_field fused data in original position in a TSDF field representation
	 * \return warped live field: live field after alignment
	 */
	virtual eig::MatrixXf optimize(const eig::MatrixXf& live_field, const eig::MatrixXf& canonical_field) = 0;

protected:
	static bool are_termination_conditions_reached(int completed_iteration_count, float largest_warp_vector);


};

}//namespace nonrigid_optimization


