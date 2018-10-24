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

//local
#include "vector2.hpp"
#include "matrix2.hpp"

namespace math{
	typedef Eigen::Matrix<math::Vector2<float>, Eigen::Dynamic, Eigen::Dynamic> MatrixXv2f;
	typedef Eigen::Matrix<math::Matrix2<float>, Eigen::Dynamic, Eigen::Dynamic> MatrixXm2f;
}



