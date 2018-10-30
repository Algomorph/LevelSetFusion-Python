//  ================================================================
//  Created by Gregory Kramida on 10/29/18.
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

//local
#include "../src/math/typedefs.hpp"
#include "../src/math/tensors.hpp"


namespace eig = Eigen;

namespace test_data{

static math::MatrixXv2f warped_live_field_gradient = []{
	eig::MatrixXf u_vectors(4,4), v_vectors(4,4);
	u_vectors << -0., -0., 0.0334751, 0.01388371,
			-0., 0.04041886, 0.0149368, 0.00573045,
			-0., 0.06464156, 0.01506416, 0.00579486,
			-0., 0.06037777, 0.0144603, 0.01164452;
	v_vectors << -0., -0., 0.019718, 0.02146172,
			-0., 0.03823357, 0.02406227, 0.02212186,
			-0., 0.02261183, 0.01864575, 0.02234527,
			-0., 0.01906347, 0.01756042, 0.02574961;
	return math::stack_as_xv2f(u_vectors, v_vectors);
}();

static eig::MatrixXf warped_live_field = []{
	eig::MatrixXf warped_live_field(4,4);
	warped_live_field <<
	                  1.f, 1.f, 0.49404836f, 0.4321034f,
			1.f, 0.44113636f, 0.34710377f, 0.32715625f,
			1.f, 0.3388706f, 0.24753733f, 0.22598255f,
			1.f, 0.21407352f, 0.16514614f, 0.11396749;
	return warped_live_field;
}();

static eig::MatrixXf canonical_field = []{
	eig::MatrixXf canonical_field(4,4);
	canonical_field << 
	        1.0000000e+00f, 1.0000000e+00f, 3.7499955e-01f, 2.4999955e-01f,
			1.0000000e+00f, 3.2499936e-01f, 1.9999936e-01f, 1.4999935e-01f,
			1.0000000e+00f, 1.7500064e-01f, 1.0000064e-01f, 5.0000645e-02f,
			1.0000000e+00f, 7.5000443e-02f, 4.4107438e-07f, -9.9999562e-02f;
	return canonical_field;
}();

}// namespace test_data