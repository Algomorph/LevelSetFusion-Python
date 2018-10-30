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
#include <Eigen/Eigen>

namespace eig = Eigen;

namespace test_data{

static eig::MatrixXf u_vectors = {}();
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
}// namespace test_data