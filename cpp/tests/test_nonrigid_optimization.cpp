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

//test targets
#include "../src/nonrigid_optimization/data_term.hpp"
#include "../src/nonrigid_optimization/interpolation.hpp"

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
	0.33603188, 0.51519966, 0.3105523 , 0.23966147,
	0.6868598 , 0.527026  , 0.48375335, 0.32714397,
	0.93489724, 0.6609843 , 0.39621043, 0.7018631 ,
	0.5436787 , 0.3114709 , 0.3591068 , 0.294315  ;
	//@formatter:on
	MatrixXf canonical_field(4, 4);
	canonical_field << //@formatter:off
	0.47082266, 0.04617875, 0.1348223 , 0.8912608,
	0.81490934, 0.30303016, 0.94416624, 0.857193 ,
	0.8719212 , 0.338506  , 0.36536142, 0.7481886,
	0.6384124 , 0.88480324, 0.35456964, 0.6872044;
	//@formatter:on
	MatrixXf live_gradient_x_field(4, 4);
	live_gradient_x_field << //@formatter:off
	 0.3508279 ,  0.01182634,  0.17320105,  0.0874825 ,
     0.2994327 ,  0.07289231,  0.04282907,  0.23110083,
    -0.07159054, -0.10777755, -0.06232327, -0.01641448,
    -0.39121854, -0.34951338, -0.03710362, -0.4075481 ;
	//@formatter:on
	MatrixXf live_gradient_y_field(4, 4);
	live_gradient_y_field << //@formatter:off
	0.17916778, -0.01273979, -0.1377691 , -0.07089083,
   -0.15983379, -0.10155322, -0.09994102, -0.15660939,
   -0.27391297, -0.2693434 ,  0.02043942,  0.30565268,
   -0.2322078 , -0.09228595, -0.00857794, -0.0647918 ;
	//@formatter:on
	int x, y;
	float expected_data_grad_x, expected_data_grad_y, expected_energy_contribution;
	float out_data_grad_x, out_data_grad_y, out_energy_contribution;
	x = 0;
	y = 0;
	data_term::data_term_at_location(warped_live_field, canonical_field, x, y,
	                                 live_gradient_x_field, live_gradient_y_field, out_data_grad_x, out_data_grad_y,
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
	MatrixXf warped_live_field(2,2), canonical_field(2,2);
	MatrixXf u_vectors(2,2), v_vectors(2,2);
	//@formatter:off
	u_vectors << 0.5F, -0.5F,
				 0.5F, -0.5F;
	v_vectors << 0.5F,  0.5F,
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
	MatrixXf warped_live_field(3,3), canonical_field(3,3);
	MatrixXf u_vectors(3,3), v_vectors(3,3);
	//@formatter:off
	u_vectors << 0.0F, 0.0F, 0.0F,
                 -.5F, 0.0F, 0.0F,
                 1.5F, 0.5F, 0.0F;
	v_vectors << -1.0F, 0.0F, 0.0F,
                  0.0F, 0.0F, 0.5F,
                 -1.5F, 0.5F,-0.5F;
	canonical_field << -1.0F, 0.F, 0.0F,
                        0.0F, 0.F, 1.0F,
                        0.0F, 0.F, 1.0F;
	warped_live_field << 1.0F, 1.0F, 1.0F,
                         0.5F, 1.0F, 1.0F,
                         0.5F, 0.5F,-1.0F;
	//@formatter:on
	Matrix3f warped_live_field_out = interpolation::interpolate(warped_live_field, canonical_field,
	                                                            u_vectors, v_vectors, true, false, true);
	Matrix3f expected_live_out;
	//@formatter:off
	expected_live_out << 1.0F,  1.000F,  1.0F,
                         0.5F,  1.000F,  1.0F,
                         1.0F,  0.125F, -1.0F;
	//@formatter:on


	BOOST_REQUIRE(warped_live_field_out.isApprox(expected_live_out));
}