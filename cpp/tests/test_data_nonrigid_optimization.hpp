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

namespace test_data {

static math::MatrixXv2f warp_field = [] {
	eig::MatrixXf u_vectors(4, 4), v_vectors(4, 4);
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

static eig::MatrixXf warped_live_field = [] {
	eig::MatrixXf warped_live_field(4, 4);
	warped_live_field <<
	                  1.f, 1.f, 0.49404836f, 0.4321034f,
			1.f, 0.44113636f, 0.34710377f, 0.32715625f,
			1.f, 0.3388706f, 0.24753733f, 0.22598255f,
			1.f, 0.21407352f, 0.16514614f, 0.11396749f;
	return warped_live_field;
}();

static eig::MatrixXf canonical_field = [] {
	eig::MatrixXf canonical_field(4, 4);
	canonical_field <<
	                -1.0000000e+00f, 1.0000000e+00f, 3.7499955e-01f, 2.4999955e-01f,
			-1.0000000e+00f, 3.2499936e-01f, 1.9999936e-01f, 1.4999935e-01f,
			1.0000000e+00f, 1.7500064e-01f, 1.0000064e-01f, 5.0000645e-02f,
			1.0000000e+00f, 7.5000443e-02f, 4.4107438e-07f, -9.9999562e-02f;
	return canonical_field;
}();

static math::MatrixXv2f data_term_gradient = [] {
	math::MatrixXv2f grad(4, 4);
	grad << math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.f, -0.f),
			math::Vector2f(-0.33803707f, -0.17493579f),
			math::Vector2f(-0.11280416f, -0.1911128f), //row 1

			math::Vector2f(-11.1772728f, 0.f),
			math::Vector2f(-0.37912705f, -0.38390793f),
			math::Vector2f(-0.08383488f, -0.1813143f),
			math::Vector2f(-0.03533841f, -0.18257865f), //row 2

			math::Vector2f(-0.f, 0.f),
			math::Vector2f(-0.61653014f, -0.18604389f),
			math::Vector2f(-0.08327565f, -0.13422713f),
			math::Vector2f(-0.03793251f, -0.18758682f), //row 3

			math::Vector2f(-0.f, 0.f),
			math::Vector2f(-0.58052848f, -0.17355914f),
			math::Vector2f(-0.0826604f, -0.13606551f),
			math::Vector2f(-0.10950545f, -0.23967532f); //row4
	return grad;
}();

static math::MatrixXv2f data_term_gradient_band_union_only = [] {
	math::MatrixXv2f grad(4, 4);
	grad << math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.f, -0.f),
			math::Vector2f(-0.33803707f, -0.17493579f),
			math::Vector2f(-0.11280416f, -0.1911128f), //row 1

			math::Vector2f(-0.0f, 0.f),
			math::Vector2f(-0.37912705f, -0.38390793f),
			math::Vector2f(-0.08383488f, -0.1813143f),
			math::Vector2f(-0.03533841f, -0.18257865f), //row 2

			math::Vector2f(-0.f, 0.f),
			math::Vector2f(-0.61653014f, -0.18604389f),
			math::Vector2f(-0.08327565f, -0.13422713f),
			math::Vector2f(-0.03793251f, -0.18758682f), //row 3

			math::Vector2f(-0.f, 0.f),
			math::Vector2f(-0.58052848f, -0.17355914f),
			math::Vector2f(-0.0826604f, -0.13606551f),
			math::Vector2f(-0.10950545f, -0.23967532f); //row4
	return grad;
}();

static math::MatrixXv2f warp_field2 = data_term_gradient_band_union_only * 0.1;

static eig::MatrixXf warped_live_field2 = [] {
	eig::MatrixXf warped_live_field(4, 4);
	warped_live_field <<
	                  1.f, 1.f, 0.51970311f, 0.44364204f,
			1.f, 0.48296618f, 0.35061902f, 0.32914556f,
			1.f, 0.38141651f, 0.24963467f, 0.22796208f,
			1.f, 0.26173902f, 0.16667641f, 0.11720487f;
	return warped_live_field;
}();

static math::MatrixXv2f tikhonov_gradient = [] {
	math::MatrixXv2f grad(4, 4);
	grad << math::Vector2f(-0.f, -0.f),
			math::Vector2f(0.07171641f, 0.05588437f),
			math::Vector2f(-0.08174722f, -0.01523803f),
			math::Vector2f(0.01477672f, -0.00247112f),

			math::Vector2f(0.0379127f, 0.03839079f),
			math::Vector2f(-0.08161432f, -0.11682735f),
			math::Vector2f(0.05004386f, 0.01503923f),
			math::Vector2f(0.01285563f, 0.0012278f),

			math::Vector2f(0.06165301f, 0.01860439f),
			math::Vector2f(-0.14231894f, -0.00524814f),
			math::Vector2f(0.04878554f, 0.0154102f),
			math::Vector2f(0.0114322f, -0.00062794f),

			math::Vector2f(0.05805285f, 0.01735591f),
			math::Vector2f(-0.10423949f, -0.0198568f),
			math::Vector2f(0.05253284f, 0.01392651f),
			math::Vector2f(-0.0098418f, -0.01556983f);
	return grad;
}();

static float tikhonov_energy = 0.00955238193f;
static math::MatrixXv2f tikhonov_gradient_band_union_only = [] {
	math::MatrixXv2f grad(4, 4);
	grad << math::Vector2f(0.f, 0.f),
			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.08174722f, -0.01523803f),
			math::Vector2f(0.01477672f, -0.00247112f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.08161432f, -0.11682735f),
			math::Vector2f(0.05004386f, 0.01503923f),
			math::Vector2f(0.01285563f, 0.0012278f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.14231893f, -0.00524814f),
			math::Vector2f(0.04878553f, 0.0154102f),
			math::Vector2f(0.0114322f, -0.00062794f),

			math::Vector2f(0.f, 0.f),
			math::Vector2f(-0.10423949f, -0.0198568f),
			math::Vector2f(0.05253284f, 0.01392651f),
			math::Vector2f(-0.0098418f, -0.01556983f);
	return grad;
}();

static float tikhonov_energy_band_union_only = 0.001989292759769734f;


}// namespace test_data