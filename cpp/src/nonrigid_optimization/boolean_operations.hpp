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

#include <Eigen/Eigen>

namespace eig = Eigen;

namespace nonrigid_optimization{
    inline bool is_outside_narrow_band_tolerance(float live_tsdf_value, float canonical_tsdf_value, float tolerance = 10e-6f){
        return (1.0f - std::abs(live_tsdf_value) < tolerance && 1.0f - std::abs(canonical_tsdf_value) < tolerance);
    }
    inline bool is_outside_narrow_band(float live_tsdf_value, float canonical_tsdf_value){
    	return std::abs(live_tsdf_value) == 1.0 && std::abs(canonical_tsdf_value) == 1.0;
    }
}// namespace boolean_ops
