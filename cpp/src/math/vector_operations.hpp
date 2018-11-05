//  ================================================================
//  Created by Gregory Kramida on 10/31/18.
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

//======================================================================================================================
//                                  Generic vector operations (Inspired by InfiniTAM ORUtils library)
//======================================================================================================================

#include "platform_independence.hpp"

namespace math{

template<typename T> _CPU_AND_GPU_CODE_ inline T squared(const T& v) { return v * v; }

// compute the dot product of two vectors
template<typename T> _CPU_AND_GPU_CODE_ inline typename T::value_type dot(const T &lhs, const T &rhs) {
	typename T::value_type r = 0;
	for (int i = 0; i < T::size; i++)
		r += lhs[i] * rhs[i];
	return r;
}

// return the length of the provided vector, L2 norm
template<typename T> _CPU_AND_GPU_CODE_ inline typename T::value_type length(const T &vec) {
	return sqrt(dot(vec, vec));
}

// return the sum of the provided vector's components
template<typename T> _CPU_AND_GPU_CODE_ inline typename T::value_type sum(const T &vec) {
	typename T::value_type r = 0;
	for (int i = 0; i < T::size; i++)
		r += vec[i];
	return r;
}

// return the sum of the squares of the provided vector's components, (L2 norm)^2
template<typename T> _CPU_AND_GPU_CODE_ inline typename T::value_type squared_sum(const T &vec) {
	typename T::value_type r = 0;
	for (int i = 0; i < T::size; i++)
		r += vec[i]*vec[i];
	return r;
}

// return the normalized version of the vector
template<typename T> _CPU_AND_GPU_CODE_ inline T normalize(const T &vec)	{
	typename T::value_type sum = length(vec);
	return sum == 0 ? T(typename T::value_type(0)) : vec / sum;
}

}//namespace math