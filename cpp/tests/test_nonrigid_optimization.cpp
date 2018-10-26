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

#define BOOST_TEST_MODULE test_nonrigid_optimization

//stdlib

//libraries
#include <boost/test/included/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//test data
#include "test_data_nonrigid_optimization.hpp"

//test targets
#include "../src/nonrigid_optimization/data_term.hpp"
#include "../src/nonrigid_optimization/smoothing_term.hpp"
#include "../src/nonrigid_optimization/interpolation.hpp"
#include "../src/math/gradients.hpp"
#include "../src/math/tensors.hpp"
#include "../src/math/typedefs.hpp"

namespace tt = boost::test_tools;
namespace bp = boost::python;

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
	data_term::compute_local_data_term_gradient(warped_live_field, canonical_field, x, y,
	                                            live_gradient_x_field, live_gradient_y_field, out_data_grad_x,
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
	Matrix2f warped_live_field_out = interpolation::interpolate(warped_live_field, canonical_field,
	                                                            u_vectors, v_vectors);
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
	Matrix3f warped_live_field_out = interpolation::interpolate(warped_live_field, canonical_field,
	                                                            u_vectors, v_vectors, true, false, true);


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
	MatrixXf warped_live_field_out = interpolation::interpolate(warped_live_field, canonical_field,
	                                                            u_vectors, v_vectors, false, false, false);

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
	MatrixXf warped_live_field(4, 4), canonical_field(4, 4);
	MatrixXf u_vectors(4, 4), v_vectors(4, 4);
	//@formatter:off
    u_vectors << -0., -0., 0.0334751, 0.01388371,
            -0., 0.04041886, 0.0149368, 0.00573045,
            -0., 0.06464156, 0.01506416, 0.00579486,
            -0., 0.06037777, 0.0144603, 0.01164452;
    v_vectors << -0., -0., 0.019718, 0.02146172,
            -0., 0.03823357, 0.02406227, 0.02212186,
            -0., 0.02261183, 0.01864575, 0.02234527,
            -0., 0.01906347, 0.01756042, 0.02574961;
    warped_live_field << 1., 1., 0.49404836, 0.4321034,
            1., 0.44113636, 0.34710377, 0.32715625,
            1., 0.3388706, 0.24753733, 0.22598255,
            1., 0.21407352, 0.16514614, 0.11396749;
    canonical_field << 1.0000000e+00, 1.0000000e+00, 3.7499955e-01, 2.4999955e-01,
            1.0000000e+00, 3.2499936e-01, 1.9999936e-01, 1.4999935e-01,
            1.0000000e+00, 1.7500064e-01, 1.0000064e-01, 5.0000645e-02,
            1.0000000e+00, 7.5000443e-02, 4.4107438e-07, -9.9999562e-02;
    //@formatter:on
	MatrixXf warped_live_field_out = interpolation::interpolate(warped_live_field, canonical_field,
	                                                            u_vectors, v_vectors, false, false, false);

	MatrixXf expected_live_out(4, 4);
	//@formatter:off
    expected_live_out << 1., 1., 0.48910502, 0.43776682,
            1., 0.43342987, 0.34440944, 0.3287866,
            1., 0.33020678, 0.24566805, 0.22797936,
            1., 0.2261582, 0.17907946, 0.14683424;
    //@formatter:on


	BOOST_REQUIRE(warped_live_field_out.isApprox(expected_live_out));
}

BOOST_AUTO_TEST_CASE(gradient_test01) {
	namespace eig = Eigen;
	eig::Matrix2f field;
	field << -0.46612028, -0.8161121,
			0.2427629, -0.79432599;


	eig::Matrix2f expected_gradient_x, expected_gradient_y;
	expected_gradient_x << -0.34999183, -0.34999183,
			-1.03708889, -1.03708889;
	expected_gradient_y << 0.70888318, 0.02178612,
			0.70888318, 0.02178612;

	eig::MatrixXf gradient_x, gradient_y;
	math::scalar_field_gradient(field, gradient_x, gradient_y);

	BOOST_REQUIRE(gradient_x.isApprox(expected_gradient_x));
	BOOST_REQUIRE(gradient_y.isApprox(expected_gradient_y));
}


BOOST_AUTO_TEST_CASE(gradient_test02) {
	using namespace Eigen;
	Matrix3f field;
	field << 0.11007435, -0.94589225, -0.54835034,
			-0.09617922, 0.15561824, 0.60624432,
			-0.83068796, 0.19262577, -0.21090505;


	Matrix3f expected_gradient_x, expected_gradient_y;
	expected_gradient_x << -1.0559666, -0.32921235, 0.39754191,
			0.25179745, 0.35121177, 0.45062608,
			1.02331373, 0.30989146, -0.40353082;
	expected_gradient_y << -0.20625357, 1.10151049, 1.15459466,
			-0.47038115, 0.56925901, 0.16872265,
			-0.73450874, 0.03700753, -0.81714937;

	MatrixXf gradient_x, gradient_y;
	math::scalar_field_gradient(field, gradient_x, gradient_y);

	BOOST_REQUIRE(gradient_x.isApprox(expected_gradient_x));
	BOOST_REQUIRE(gradient_y.isApprox(expected_gradient_y));
}

BOOST_AUTO_TEST_CASE(gradient_test03) {
	using namespace Eigen;

	MatrixXf gradient_x, gradient_y;
	math::scalar_field_gradient(test_data::field, gradient_x, gradient_y);

	BOOST_REQUIRE(gradient_x.isApprox(test_data::expected_gradient_x));
	BOOST_REQUIRE(gradient_y.isApprox(test_data::expected_gradient_y));
}


BOOST_AUTO_TEST_CASE(gradient_test04) {
	namespace eig = Eigen;

	eig::Matrix2f field;
	field << -0.46612028, -0.8161121,
			0.2427629, -0.79432599;

	math::MatrixXv2f expected_gradient(2, 2);
	expected_gradient <<
	                  //@formatter:off
            math::Vector2f(-0.34999183,0.70888318), math::Vector2f(-0.34999183,0.02178612),
		    math::Vector2f(-1.03708889,0.70888318), math::Vector2f(-1.03708889,0.02178612);
    //@formatter:on

	math::MatrixXv2f gradient;
	math::scalar_field_gradient(field, gradient);

	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}


BOOST_AUTO_TEST_CASE(gradient_test05) {
	namespace eig = Eigen;

	eig::Matrix3f field;
	field << 0.11007435, -0.94589225, -0.54835034,
			-0.09617922, 0.15561824, 0.60624432,
			-0.83068796, 0.19262577, -0.21090505;

	math::MatrixXv2f expected_gradient(3, 3);
	expected_gradient <<
	                  //@formatter:off
            math::Vector2f(-1.0559666,-0.20625357), math::Vector2f(-0.32921235,1.10151049), math::Vector2f(0.39754191,1.15459466),
			math::Vector2f(0.25179745,-0.47038115), math::Vector2f(0.35121177,0.56925901), math::Vector2f(0.45062608,0.16872265),
		    math::Vector2f(1.02331373,-0.73450874), math::Vector2f(0.30989146,0.03700753), math::Vector2f(-0.40353082,-0.81714937);
    //@formatter:on
	math::MatrixXv2f gradient;
	math::scalar_field_gradient(field, gradient);

	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(gradient_test06) {
	namespace eig = Eigen;

	math::MatrixXv2f gradient;
	math::scalar_field_gradient(test_data::field, gradient);

	eig::MatrixXf exp_grad_x = test_data::expected_gradient_x;
	eig::MatrixXf exp_grad_y = test_data::expected_gradient_y;

	math::MatrixXv2f expected_gradient = math::stack_as_xv2f(test_data::expected_gradient_x,
	                                                         test_data::expected_gradient_y);
	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(vector_field_gradient_test01) {
	math::MatrixXv2f vector_field(2, 2);
	vector_field << //@formatter:off
	        math::Vector2f(0.0f, 0.0f), math::Vector2f(1.0f, -1.0f),
			math::Vector2f(-1.0f, 1.0f), math::Vector2f(1.0f, 1.0f);
	//@formatter:on

	math::MatrixXm2f gradient;
	math::vector_field_gradient(vector_field, gradient);

	math::MatrixXm2f expected_gradient(2,2);
	expected_gradient << math::Matrix2f(1.0f, -1.0f, -1.0f, 1.0f), math::Matrix2f(1.0f, 0.0f, -1.0f, 2.0f),
						 math::Matrix2f(2.0f, -1.0f, 0.0f,  1.0f), math::Matrix2f(2.0f, 0.0f, 0.0f,  2.0f);

	BOOST_REQUIRE(math::almost_equal(gradient, expected_gradient, 1e-6));
}

BOOST_AUTO_TEST_CASE(vector_field_gradient_test02) {
	math::MatrixXm2f gradient;
	math::vector_field_gradient(test_data::vector_field, gradient);
	BOOST_REQUIRE(math::almost_equal(gradient, test_data::vector_field_gradient, 1e-6));
}