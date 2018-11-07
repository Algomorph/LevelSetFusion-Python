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

//stdlib
#include <vector>
#include <cfloat>

//local
#include "sobolev_optimizer2d.hpp"
#include "data_term.hpp"
#include "smoothing_term.hpp"
#include "interpolation.hpp"
#include "max_norm.hpp"
#include "../math/gradients.hpp"
#include "../math/convolution.hpp"


namespace nonrigid_optimization {



void SobolevOptimizer2d::SobolevParameters::set_from_json(pt::ptree root) {
	this->smoothing_term_weight = root.get<float>("smoothing_term_weight", 0.2);
	std::vector<float> kernel_values;
	for (pt::ptree::value_type& value : root.get_child("sobolev_kernel")) {
		kernel_values.push_back(value.second.get_value<float>());
	}
	this->sobolev_kernel = eig::VectorXf(kernel_values.size());
	int i_value = 0;
	for (float element : kernel_values) {
		this->sobolev_kernel(i_value) = element;
		i_value++;
	}
}

Optimizer2d::SharedParameters& SobolevOptimizer2d::shared_parameters(){
	return Optimizer2d::SharedParameters::get_instance();
}

SobolevOptimizer2d::SobolevParameters& SobolevOptimizer2d::sobolev_parameters() {
	return SobolevParameters::get_instance();
}


eig::MatrixXf SobolevOptimizer2d::optimize(const eig::MatrixXf& live_field, const eig::MatrixXf& canonical_field) {
	float maximum_warp_length = SobolevOptimizer2d::shared_parameters().maximum_warp_length_upper_threshold-1.0f;

	eig::MatrixXf warped_live_field = live_field;
	math::MatrixXv2f warp_field = math::MatrixXv2f::Zero(live_field.rows(), live_field.cols());

	for (int completed_iteration_count = 0;
	     !Optimizer2d::are_termination_conditions_reached(completed_iteration_count, maximum_warp_length);
	     completed_iteration_count++) {
		maximum_warp_length =
				perform_optimization_iteration_and_return_max_warp(warped_live_field, canonical_field, warp_field);
	}

	return warped_live_field;
}


float SobolevOptimizer2d::perform_optimization_iteration_and_return_max_warp(eig::MatrixXf& warped_live_field,
                                                                             const eig::MatrixXf& canonical_field,
                                                                             math::MatrixXv2f& warp_field) {
	math::MatrixXv2f data_term_gradient, smoothing_term_gradient, warped_live_field_gradient;
	float data_term_energy, smoothing_term_energy;
	math::scalar_field_gradient(warped_live_field, warped_live_field_gradient);
	compute_data_term_gradient_within_band_union(data_term_gradient, data_term_energy, warped_live_field,
	                                             canonical_field, warped_live_field_gradient);
	compute_tikhonov_regularization_gradient_within_band_union(smoothing_term_gradient, smoothing_term_energy,
	                                                           warp_field, warped_live_field, canonical_field);
	warp_field = (data_term_gradient + smoothing_term_gradient * sobolev_parameters().smoothing_term_weight)
	             * -shared_parameters().gradient_descent_rate;
	std::cout << "Pre-convo:" << std::endl;
	std::cout << warp_field << std::endl;
	math::convolve_with_kernel_preserve_zeros(warp_field, sobolev_parameters().sobolev_kernel);
	std::cout << "Post-convo:" << std::endl;
	std::cout << warp_field << std::endl;
	warped_live_field = interpolate(warp_field, warped_live_field, canonical_field, true);
	float maximum_warp_length; math::Vector2i maximum_warp_location;
	locate_max_norm(maximum_warp_length, maximum_warp_location, warp_field);
	return maximum_warp_length;
}
}//namespace nonrigid_optimization