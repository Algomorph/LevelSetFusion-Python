//  ================================================================
//  Created by Gregory Kramida on 11/2/18.
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

//stdlib
#include <vector>
//local
#include "sobolev_optimizer.hpp"


namespace nonrigid_optimization{


void SobolevOptimizer2d::SobolevParameters::set_from_json(pt::ptree root) {
	this->smoothing_term_weight = root.get<float>("smoothing_term_weight", 0.2);
	std::vector<float> kernel_values;
	for(pt::ptree::value_type & value : root.get_child("sobolev_kernel")){
		kernel_values.push_back(value.second.get_value<float>());
	}
	this->sobolev_kernel = eig::VectorXf(kernel_values.size());
	int i_value = 0;
	for (float element : kernel_values){
		this->sobolev_kernel(i_value) = element;
		i_value++;
	}
}


}//namespace nonrigid_optimization