//  ================================================================
//  Created by Gregory Kramida on 10/10/18.
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

#include <iostream>

#define BOOST_TEST_MODULE test_nonrigid_optimization // NB:has to appear before the boost include

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//local
#include "../src/math/gradients.hpp"

//test data
#include "test_data_nonrigid_optimization.hpp"

//test targets
#include "../src/nonrigid_optimization/data_term.hpp"
#include "../src/nonrigid_optimization/smoothing_term.hpp"
#include "../src/nonrigid_optimization/interpolation.hpp"
#include "../src/nonrigid_optimization/sobolev_optimizer2d.hpp"

namespace tt = boost::test_tools;
namespace bp = boost::python;
namespace eig = Eigen;
namespace no = nonrigid_optimization;

BOOST_AUTO_TEST_CASE(test_test) {

	int a = 1;
	int b = 2;
	BOOST_REQUIRE(a != b - 2);
	BOOST_REQUIRE(a + 1 <= b);
}

BOOST_AUTO_TEST_CASE(data_term_test) {
	using namespace Eigen;

	MatrixXf warped_live_field(4, 4);
	// some pre-computed test data

	warped_live_field << //@formatter:off
            0.33603188, 0.51519966, 0.3105523, 0.23966147,
            0.6868598, 0.527026, 0.48375335, 0.32714397,
            0.93489724, 0.6609843, 0.39621043, 0.7018631,
            0.5436787, 0.3114709, 0.3591068, 0.294315;
    //@formatter:on
	MatrixXf canonical_field(4, 4);
	canonical_field << //@formatter:off
             0.47082266, 0.04617875, 0.1348223, 0.8912608,
            0.81490934, 0.30303016, 0.94416624, 0.857193,
            0.8719212, 0.338506, 0.36536142, 0.7481886,
            0.6384124, 0.88480324, 0.35456964, 0.6872044;
    //@formatter:on
	MatrixXf live_gradient_x_field(4, 4);
	live_gradient_x_field << //@formatter:off
            0.3508279, 0.01182634, 0.17320105, 0.0874825,
            0.2994327, 0.07289231, 0.04282907, 0.23110083,
            -0.07159054, -0.10777755, -0.06232327, -0.01641448,
            -0.39121854, -0.34951338, -0.03710362, -0.4075481;
    //@formatter:on
	MatrixXf live_gradient_y_field(4, 4);
	live_gradient_y_field << //@formatter:off
            0.17916778, -0.01273979, -0.1377691, -0.07089083,
            -0.15983379, -0.10155322, -0.09994102, -0.15660939,
            -0.27391297, -0.2693434, 0.02043942, 0.30565268,
            -0.2322078, -0.09228595, -0.00857794, -0.0647918;
    //@formatter:on
	int x, y;
	float expected_data_grad_x, expected_data_grad_y, expected_energy_contribution;
	float out_data_grad_x, out_data_grad_y, out_energy_contribution;
	x = 0;
	y = 0;
	nonrigid_optimization::compute_local_data_term_gradient(warped_live_field, canonical_field, x, y,
	                                                        live_gradient_x_field, live_gradient_y_field,
	                                                        out_data_grad_x,
	                                                        out_data_grad_y,
	                                                        out_energy_contribution);

	expected_data_grad_x = -0.47288364F;
	expected_data_grad_y = -0.24150164F;
	expected_energy_contribution = 0.009084276938502F;

	BOOST_REQUIRE_CLOSE(out_data_grad_x, expected_data_grad_x, 10e-6);
	BOOST_REQUIRE_CLOSE(out_data_grad_y, expected_data_grad_y, 10e-6);
	BOOST_REQUIRE_CLOSE(out_energy_contribution, expected_energy_contribution, 10e-6);
}

BOOST_AUTO_TEST_CASE(interpolation_test01) {
	using namespace Eigen;
	MatrixXf warped_live_field(2, 2), canonical_field(2, 2);
	MatrixXf u_vectors(2, 2), v_vectors(2, 2);
	//@formatter:off
    u_vectors << 0.5F, -0.5F,
            0.5F, -0.5F;
    v_vectors << 0.5F, 0.5F,
            -0.5F, -0.5F;
    canonical_field << 0.0F, 0.0F,
            0.0F, 0.0F;
    warped_live_field << 1.0F, -1.0F,
            1.0F, -1.0F;
    //@formatter:on
	math::MatrixXv2f warp_field = math::stack_as_xv2f(u_vectors, v_vectors);
	Matrix2f warped_live_field_out = nonrigid_optimization::interpolate(warp_field, warped_live_field, canonical_field);
	Matrix2f expected_live_out;
	expected_live_out << 0.0F, 0.0F, 0.0F, 0.0F;

	BOOST_REQUIRE(warped_live_field_out.isApprox(expected_live_out));
}

BOOST_AUTO_TEST_CASE(interpolation_test02) {
	using namespace Eigen;
	MatrixXf warped_live_field(3, 3), canonical_field(3, 3);
	MatrixXf u_vectors(3, 3), v_vectors(3, 3);
	//@formatter:off
    u_vectors << 0.0F, 0.0F, 0.0F,
            -.5F, 0.0F, 0.0F,
            1.5F, 0.5F, 0.0F;
    v_vectors << -1.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 0.5F,
            -1.5F, 0.5F, -0.5F;
    canonical_field << -1.0F, 0.F, 0.0F,
            0.0F, 0.F, 1.0F,
            0.0F, 0.F, 1.0F;
    warped_live_field << 1.0F, 1.0F, 1.0F,
            0.5F, 1.0F, 1.0F,
            0.5F, 0.5F, -1.0F;
    //@formatter:on
	math::MatrixXv2f warp_field = math::stack_as_xv2f(u_vectors, v_vectors);
	Matrix3f warped_live_field_out = nonrigid_optimization::interpolate(warp_field, warped_live_field, canonical_field,
	                                                                    true, false, true);


	Matrix3f expected_live_out;
	//@formatter:off
    expected_live_out << 1.0F, 1.000F, 1.0F,
            0.5F, 1.000F, 1.0F,
            1.0F, 0.125F, -1.0F;
    //@formatter:on

	BOOST_REQUIRE(warped_live_field_out.isApprox(expected_live_out));
}

BOOST_AUTO_TEST_CASE(interpolation_test03) {
	using namespace Eigen;
	MatrixXf warped_live_field(4, 4), canonical_field(4, 4);
	MatrixXf u_vectors(4, 4), v_vectors(4, 4);
	//@formatter:off
    u_vectors << -0., -0., 0.03732542, 0.01575381,
            -0., 0.04549519, 0.01572882, 0.00634488,
            -0., 0.07203466, 0.01575179, 0.00622413,
            -0., 0.05771814, 0.01468342, 0.01397111;
    v_vectors << -0., -0., 0.02127664, 0.01985903,
            -0., 0.04313552, 0.02502393, 0.02139519,
            -0., 0.02682102, 0.0205336, 0.02577237,
            -0., 0.02112256, 0.01908935, 0.02855439;
    warped_live_field << 1., 1., 0.49999955, 0.42499956,
            1., 0.44999936, 0.34999937, 0.32499936,
            1., 0.35000065, 0.25000066, 0.22500065,
            1., 0.20000044, 0.15000044, 0.07500044;
    canonical_field << 1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01,
            1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01,
            1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02,
            1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02;
    //@formatter:on
	math::MatrixXv2f warp_field = math::stack_as_xv2f(u_vectors, v_vectors);
	MatrixXf warped_live_field_out = nonrigid_optimization::interpolate(warp_field, warped_live_field, canonical_field,
	                                                                    false, false, false);

	MatrixXf expected_live_out(4, 4);
	//@formatter:off
    expected_live_out << 1., 1., 0.49404836, 0.4321034,
            1., 0.44113636, 0.34710377, 0.32715625,
            1., 0.3388706, 0.24753733, 0.22598255,
            1., 0.21407352, 0.16514614, 0.11396749;
    //@formatter:on

	BOOST_REQUIRE(warped_live_field_out.isApprox(expected_live_out));
}

BOOST_AUTO_TEST_CASE(interpolation_test04) {
	using namespace Eigen;

	MatrixXf warped_live_field_out = nonrigid_optimization::interpolate(
			test_data::warp_field, test_data::warped_live_field, test_data::canonical_field, false, false, false);

	MatrixXf expected_live_out(4, 4);
	//@formatter:off
    expected_live_out << 1., 1., 0.48910502, 0.43776682,
            1., 0.43342987, 0.34440944, 0.3287866,
            1., 0.33020678, 0.24566805, 0.22797936,
            1., 0.2261582, 0.17907946, 0.14683424;
    //@formatter:on

	BOOST_REQUIRE(warped_live_field_out.isApprox(expected_live_out));
}

BOOST_AUTO_TEST_CASE(test_data_term_gradient01) {
	math::MatrixXv2f data_term_gradient, data_term_gradient_band_union_only;
	float data_term_energy;
	math::MatrixXv2f warped_live_field_gradient;
	math::scalar_field_gradient(test_data::warped_live_field, warped_live_field_gradient);
	float expected_energy_out = 4.142916451210006f;
	float expected_energy_out_band_union_only = 0.14291645121000718f;
	nonrigid_optimization::compute_data_term_gradient(data_term_gradient, data_term_energy,
	                                                  test_data::warped_live_field, test_data::canonical_field,
	                                                  warped_live_field_gradient);
	BOOST_REQUIRE(math::almost_equal(data_term_gradient, test_data::data_term_gradient, 1e-6));
	BOOST_REQUIRE_CLOSE(data_term_energy,expected_energy_out,1e-6);
	nonrigid_optimization::compute_data_term_gradient_within_band_union(data_term_gradient_band_union_only,
	                                                                    data_term_energy, test_data::warped_live_field,
	                                                                    test_data::canonical_field,
	                                                                    warped_live_field_gradient);
	BOOST_REQUIRE(math::almost_equal(data_term_gradient_band_union_only,
	                                 test_data::data_term_gradient_band_union_only, 1e-6));
	BOOST_REQUIRE_CLOSE(data_term_energy,expected_energy_out_band_union_only,1e-6);

}

BOOST_AUTO_TEST_CASE(test_tikhonov_regularization_gradient01) {
	eig::MatrixXf live_field(2, 2), canonical_field(2, 2);
	math::MatrixXv2f warp_field(2, 2);
	math::MatrixXv2f tikhonov_gradient_band_union_only;
	float tikhonov_energy_band_union_only;
	math::MatrixXv2f expected_gradient_out(2, 2);

	// Note on expected energy values: this value assumes that the gradient at the borders is computed as forward/back
	// finite differences, NOT as central differences with border value replication. For the latter case,
	// the energy value would be 0.375, see compute_smoothing_term_gradient_direct in the smoothing_term module of the
	// python codebase.
	float expected_energy_out = 1.5;

	//@formatter:off
	live_field <<
	         0.0f, 1.0f,
			-1.0f, 0.5f;
	canonical_field <<
	        1.0f, -1.0f,
			0.5f, 0.5f;
	warp_field <<
			math::Vector2f(0.0f,0.0f),   math::Vector2f(0.5f,0.5f),
			math::Vector2f(-0.5f,-0.5f), math::Vector2f(0.0f,0.f);
	expected_gradient_out <<
			math::Vector2f(0.0f,0.0f), math::Vector2f(0.0f,0.0f),
			math::Vector2f(-1.f,-1.f), math::Vector2f(0.0f,0.0f);
	//@formatter:on

	nonrigid_optimization::compute_tikhonov_regularization_gradient_within_band_union(
			tikhonov_gradient_band_union_only, tikhonov_energy_band_union_only,
			warp_field, live_field, canonical_field);

	BOOST_REQUIRE(math::almost_equal_verbose(tikhonov_gradient_band_union_only, expected_gradient_out, 1e-6));
	BOOST_REQUIRE_CLOSE(tikhonov_energy_band_union_only, expected_energy_out, 1e-6);
}

BOOST_AUTO_TEST_CASE(test_tikhonov_regularization_gradient02) {
	math::MatrixXv2f tikhonov_gradient, tikhonov_gradient_band_union_only;
	float tikhonov_energy;

	math::MatrixXv2f warp_field = test_data::data_term_gradient_band_union_only * 0.1;

	nonrigid_optimization::compute_tikhonov_regularization_gradient(
			tikhonov_gradient, tikhonov_energy, warp_field);

	BOOST_REQUIRE(math::almost_equal_verbose(tikhonov_gradient, test_data::tikhonov_gradient, 1e-6));
	BOOST_REQUIRE_CLOSE(tikhonov_energy, test_data::tikhonov_energy, 1e-4);

	nonrigid_optimization::compute_tikhonov_regularization_gradient_within_band_union(
			tikhonov_gradient_band_union_only, tikhonov_energy, warp_field, test_data::warped_live_field2,
			test_data::canonical_field);

	BOOST_REQUIRE(
			math::almost_equal_verbose(tikhonov_gradient_band_union_only, test_data::tikhonov_gradient_band_union_only,
			                           1e-6));
	BOOST_REQUIRE_CLOSE(tikhonov_energy, test_data::tikhonov_energy_band_union_only, 1e-4);
}

BOOST_AUTO_TEST_CASE(test_sobolev_optimizer01) {
	//corresponds to Python test_nonrigid_optimization01 in test_nonrigid_optimization.py
	eig::Vector3f sobolev_kernel;
	sobolev_kernel << 0.06742075f, 0.99544406f, 0.06742075f;
	no::SobolevOptimizer2d::shared_parameters().maximum_iteration_count = 1;
	no::SobolevOptimizer2d::sobolev_parameters().sobolev_kernel = sobolev_kernel;
	no::SobolevOptimizer2d optimizer;
	eig::MatrixXf live_field(4, 4), canonical_field(4, 4);
	live_field << 1.f, 1.f, 0.49999955f, 0.42499956f,
			1.f, 0.44999936f, 0.34999937f, 0.32499936f,
			1.f, 0.35000065f, 0.25000066f, 0.22500065f,
			1.f, 0.20000044f, 0.15000044f, 0.07500044f;
	canonical_field << 1.0000000e+00f, 1.0000000e+00f, 3.7499955e-01f, 2.4999955e-01f,
			1.0000000e+00f, 3.2499936e-01f, 1.9999936e-01f, 1.4999935e-01f,
			1.0000000e+00f, 1.7500064e-01f, 1.0000064e-01f, 5.0000645e-02f,
			1.0000000e+00f, 7.5000443e-02f, 4.4107438e-07f, -9.9999562e-02f;

	eig::MatrixXf expected_warped_live_field_out(4, 4);
	expected_warped_live_field_out <<
			//@formatter:off
			1.0f, 1.0f,        0.49408937f, 0.4321034f,
			1.0f, 0.44113636f, 0.34710377f, 0.32715625f,
			1.0f, 0.3388706f,  0.24753733f, 0.22598255f,
			1.0f, 0.21407352f, 0.16514614f, 0.11396749f;
	//@formatter: on

	eig::MatrixXf warped_live_field = optimizer.optimize(live_field, canonical_field);

	BOOST_REQUIRE(warped_live_field.isApprox(expected_warped_live_field_out));
}

BOOST_AUTO_TEST_CASE(test_sobolev_optimizer02) {
	//corresponds to Python test_nonrigid_optimization02 in test_nonrigid_optimization.py
	eig::Vector3f sobolev_kernel;
	sobolev_kernel << 0.06742075f, 0.99544406f, 0.06742075f;
	no::SobolevOptimizer2d::shared_parameters().maximum_iteration_count = 2;
	no::SobolevOptimizer2d::shared_parameters().maximum_warp_length_lower_threshold = 0.05f;
	no::SobolevOptimizer2d::sobolev_parameters().sobolev_kernel = sobolev_kernel;
	no::SobolevOptimizer2d optimizer;
	eig::MatrixXf live_field(4, 4), canonical_field(4, 4);
	live_field << 1.f, 1.f, 0.49999955f, 0.42499956f,
			1.f, 0.44999936f, 0.34999937f, 0.32499936f,
			1.f, 0.35000065f, 0.25000066f, 0.22500065f,
			1.f, 0.20000044f, 0.15000044f, 0.07500044f;
	canonical_field << 1.0000000e+00f, 1.0000000e+00f, 3.7499955e-01f, 2.4999955e-01f,
			1.0000000e+00f, 3.2499936e-01f, 1.9999936e-01f, 1.4999935e-01f,
			1.0000000e+00f, 1.7500064e-01f, 1.0000064e-01f, 5.0000645e-02f,
			1.0000000e+00f, 7.5000443e-02f, 4.4107438e-07f, -9.9999562e-02f;

	eig::MatrixXf expected_warped_live_field_out(4, 4);
	expected_warped_live_field_out <<
	                               //@formatter:off
			1.0f, 1.0f,        0.48917317f, 0.43777004f,
			1.0f, 0.43342987f, 0.3444094f,  0.3287867f,
			1.0f, 0.33020678f, 0.24566807f, 0.22797936f,
			1.0f, 0.2261582f,  0.17907946f, 0.14683424f;
	//@formatter: on

	eig::MatrixXf warped_live_field = optimizer.optimize(live_field, canonical_field);

	BOOST_REQUIRE(warped_live_field.isApprox(expected_warped_live_field_out));
}
