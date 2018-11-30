//  ================================================================
//  Created by Gregory Kramida on 10/25/18.
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

#include "tensors.hpp"
#include "typedefs.hpp"

namespace math {

MatrixXv2f stack_as_xv2f(const Eigen::MatrixXf& matrix_a, const Eigen::MatrixXf& matrix_b) {
	eigen_assert((matrix_a.rows() == matrix_b.rows() && matrix_a.cols() == matrix_b.cols()) && // @suppress("Invalid arguments")
	             "Argument matrices do not have the same dimensions.");
	MatrixXv2f out(matrix_a.rows(), matrix_a.cols());
	for (Eigen::Index i_element = 0; i_element < out.size(); i_element++){
		out(i_element) = Vector2f(matrix_a(i_element), matrix_b(i_element));
	}
	return out;
}

void unstack_xv2f(Eigen::MatrixXf& matrix_a, Eigen::MatrixXf& matrix_b, const MatrixXv2f vector_field){
	matrix_a = Eigen::MatrixXf(vector_field.rows(), vector_field.cols());
	matrix_b = Eigen::MatrixXf(vector_field.rows(), vector_field.cols());
	for (Eigen::Index i_element = 0; i_element < vector_field.size(); i_element++){
		matrix_a(i_element) = vector_field(i_element).x;
		matrix_b(i_element) = vector_field(i_element).y;
	}
}

}//end namespace math
