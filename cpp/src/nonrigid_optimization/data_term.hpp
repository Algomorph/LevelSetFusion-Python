//  ================================================================
//  Created by Gregory Kramida on 10/9/18.
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
//stdlib

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <unsupported/Eigen/NumericalDiff>

//local
#include <eigen_numpy.hpp>
#include "../math/tensors.hpp"

namespace bp = boost::python;
namespace eig = Eigen;

namespace nonrigid_optimization {

void compute_data_term_gradient(
		math::MatrixXv2f& data_term_gradient, float& data_term_energy,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		const math::MatrixXv2f& warped_live_field_gradient,
		float scaling_factor = 10.0f);

void compute_data_term_gradient_within_band_union(
		math::MatrixXv2f& data_term_gradient, float& data_term_energy,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		const math::MatrixXv2f& warped_live_field_gradient,
		float scaling_factor = 10.0f);



void compute_local_data_term_gradient(const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
                                      int x, int y,
                                      const eig::MatrixXf& live_gradient_x_field,
                                      const eig::MatrixXf& live_gradient_y_field,
                                      float& data_gradient_x, float& data_gradient_y, float& local_energy_contribution);
bp::tuple py_data_term_at_location(eig::MatrixXf warped_live_field, eig::MatrixXf canonical_field, int i_col, int i_row,
                                   eig::MatrixXf live_gradient_x, eig::MatrixXf live_gradient_y);
}//namespace data_term




