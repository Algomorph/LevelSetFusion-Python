//  ================================================================
//  Created by Gregory Kramida on 11/3/18.
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

//librarires
#include <Eigen/Eigen>

//local
#include "tensors.hpp"

namespace eig = Eigen;

namespace math{
	void convolve_with_kernel_y(MatrixXv2f& field, const eig::VectorXf& kernel_1d);
	void convolve_with_kernel_x(MatrixXv2f& field, const eig::VectorXf& kernel_1d);
	void convolve_with_kernel_preserve_zeros(MatrixXv2f& field, const eig::VectorXf& kernel_1d);

}//namespace math