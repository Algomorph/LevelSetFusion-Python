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

//libraries
#include <Eigen/Eigen>
#include <boost/property_tree/ptree.hpp>


//local
#include "optimizer2d.hpp"
#include "../math/tensors.hpp"

namespace pt = boost::property_tree;
namespace eig = Eigen;

namespace nonrigid_optimization {
class SobolevOptimizer2d : public Optimizer2d {
public:
	class SobolevParameters {
	public:
		static SobolevParameters& get_instance() {
			static SobolevParameters instance;
			return instance;
		}

		SobolevParameters(const SobolevParameters&) = delete;
		void operator=(SobolevParameters const&) = delete;

		void set_from_json(pt::ptree root);

		eig::VectorXf sobolev_kernel = [] {
			eig::VectorXf sobolev_kernel(7);
			sobolev_kernel << 2.995900285895913839e-04f,
					4.410949535667896271e-03f,
					6.571318954229354858e-02f,
					9.956527948379516602e-01f,
					6.571318954229354858e-02f,
					4.410949535667896271e-03f,
					2.995900285895913839e-04f;
			return sobolev_kernel;
		}();

		float smoothing_term_weight = 0.2f;

	private:
		SobolevParameters() = default;
	};

	static SobolevParameters& sobolev_parameters();
	static SharedParameters& shared_parameters();

	virtual eig::MatrixXf optimize(const eig::MatrixXf& live_field, const eig::MatrixXf& canonical_field) override;

private:
	float perform_optimization_iteration_and_return_max_warp(eig::MatrixXf& warped_live_field,
	                                                         const eig::MatrixXf& canonical_field,
	                                                         math::MatrixXv2f& warp_field);
};

}//namespace nonrigid_optimization



