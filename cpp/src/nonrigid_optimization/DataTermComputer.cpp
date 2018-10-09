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
#include "DataTermComputer.hpp"

namespace data_term {
bp::tuple data_term_at_location(eig::MatrixXf warped_live_field, eig::MatrixXf canonical_field, int x, int y,
                                eig::MatrixXf live_gradient_x_field, eig::MatrixXf live_gradient_y_field) {
	double live_sdf = warped_live_field(y, x);
	double canonical_sdf = canonical_field(y, x);
	double difference = live_sdf - canonical_sdf;
	double scaling_factor = 10.0;
	double gradient_x = live_gradient_x_field(y, x);
	double gradient_y = live_gradient_y_field(y, x);

	eig::RowVector2f data_gradient;
	data_gradient(0) = difference * gradient_x * scaling_factor;
	data_gradient(1) = difference * gradient_y * scaling_factor;

	bp::object data_gradient_out(data_gradient);

	double local_energy_contribution = 0.5 * difference * difference;
	return bp::make_tuple(data_gradient_out, local_energy_contribution);
}
}//namespace data_term