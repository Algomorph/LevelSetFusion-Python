//  ================================================================
//  Created by Gregory Kramida on 10/9/18.
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
//stdlib

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>

//local
#include <eigen_numpy.hpp>

namespace bp = boost::python;
namespace eig = Eigen;

namespace data_term {
void data_term_at_location(const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field, int x, int y,
                           const eig::MatrixXf& live_gradient_x_field, const eig::MatrixXf& live_gradient_y_field,
                           float& data_gradient_x, float& data_gradient_y, float& local_energy_contribution);
bp::tuple py_data_term_at_location(eig::MatrixXf warped_live_field, eig::MatrixXf canonical_field, int x, int y,
                                   eig::MatrixXf live_gradient_x, eig::MatrixXf live_gradient_y);
}//namespace data_term




