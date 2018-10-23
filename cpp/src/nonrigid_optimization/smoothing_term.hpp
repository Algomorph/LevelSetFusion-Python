//  ================================================================
//  Created by Gregory Kramida on 10/23/18.
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
#include <unsupported/Eigen/CXX11/Tensor>

namespace eig = Eigen;

namespace smoothing_term {

//void gradient(const eig::MatrixXf& warp_field_x, const eig::MatrixXf& warp_field_y,
//              eig::MatrixXf& live_gradient_ux, eig::MatrixXf& live_gradient_uy,
//              eig::MatrixXf& live_gradient_vx, eig::MatrixXf& live_gradient_vy);
void gradient(const eig::Tensor<float, 3>& warp_field,eig::Tensor<float, 3>& warp_field_gradient);

void
compute_local_smoothing_term_gradient_tikhonov(
		int i_col, int i_row,
		float& smoothing_gradient_x, float& smoothing_gradient_y,
		bool ignore_if_zero = false, bool copy_if_zero = false) {

}

}//smoothing_term
