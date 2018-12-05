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
#include "../math/tensors.hpp"

namespace eig = Eigen;

namespace nonrigid_optimization {

void
compute_tikhonov_regularization_gradient(math::MatrixXv2f& gradient, float& energy, const math::MatrixXv2f& warp_field);

void
compute_tikhonov_regularization_gradient_within_band_union(math::MatrixXv2f& gradient, float& energy,
                                                           const math::MatrixXv2f& warp_field,
                                                           const eig::MatrixXf& live_field,
                                                           const eig::MatrixXf& canonical_field);

}//nonrigid_optimization
