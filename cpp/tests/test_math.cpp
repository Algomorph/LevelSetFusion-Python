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

#define BOOST_TEST_MODULE test_math

//libraries
#include <boost/test/unit_test.hpp>
#include <boost/python.hpp>
#include <Eigen/Eigen>

//test data
#include "test_data_math.hpp"

//test targets
#include "../src/math/gradients.hpp"
#include "../src/math/tensors.hpp"
#include "../src/math/typedefs.hpp"

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
            math::Vector2f(-0.34999183f,0.70888318f), math::Vector2f(-0.34999183f,0.02178612f),
		    math::Vector2f(-1.03708889f,0.70888318f), math::Vector2f(-1.03708889f,0.02178612f);
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
            math::Vector2f(-1.0559666f,-0.20625357f), math::Vector2f(-0.32921235f,1.10151049f), math::Vector2f(0.39754191f,1.15459466f),
			math::Vector2f(0.25179745f,-0.47038115f), math::Vector2f(0.35121177f,0.56925901f), math::Vector2f(0.45062608f,0.16872265f),
		    math::Vector2f(1.02331373f,-0.73450874f), math::Vector2f(0.30989146f,0.03700753f), math::Vector2f(-0.40353082f,-0.81714937f);
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