//  ================================================================
//  Created by Gregory Kramida on 11/5/18.
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

#include "max_norm.hpp"
#include "../math/vector_operations.hpp"
#include <cfloat>


namespace nonrigid_optimization{

void locate_max_norm(float& max_norm, math::Vector2i coordinate, const math::MatrixXv2f& vector_field){
	float max_squared_norm = 0;
	coordinate = math::Vector2i(0);
	int column_count = static_cast<int>(vector_field.cols());
	max_norm = 0.0f;
#pragma omp parallel for
	for(eig::Index i_element = 0; i_element < vector_field.size(); i_element++){
		float squared_length = math::squared_sum(vector_field(i_element));
		if(squared_length > max_squared_norm){
			max_squared_norm = squared_length;
			div_t division_result = div(static_cast<int>(i_element), column_count);
			coordinate.x = division_result.quot;
			coordinate.y = division_result.rem;
		}
	}
	max_norm = std::sqrt(max_squared_norm);
}


}//nonrigid_optimization
