//  ================================================================
//  Created by Gregory Kramida on 10/25/18.
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

#include "vector2.hpp"

////===================   BOOST CONVERTERS     =======================================================
//namespace math {
//struct Vector2ToFloatPointerConverter {
//	static float* convert(const Vector2& vec);
//};
//
//struct Vector2FromFloatPointerConverter {
//
//	Vector2FromFloatPointerConverter();
//
//	/// @brief Check if PyObject is an array and can be converted to OpenCV matrix.
//	static void* convertible(float* array);
//
//	/// @brief Construct a Mat from an NDArray object.
//	static void construct(float* object,
//	                      boost::python::converter::rvalue_from_python_stage1_data* data);
//};
//}// namespace math