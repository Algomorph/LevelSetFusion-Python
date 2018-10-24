//  ================================================================
//  Created by Gregory Kramida on 10/24/18.
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

//local
#include "vector2.hpp"
#include "vector3.hpp"
#include "matrix2.hpp"

namespace math{
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

typedef class math::Vector2<short> Vector2s;
typedef class math::Vector2<int> Vector2i;
typedef class math::Vector2<float> Vector2f;
typedef class math::Vector2<double> Vector2d;

typedef class math::Vector3<short> Vector3s;
typedef class math::Vector3<double> Vector3d;
typedef class math::Vector3<int> Vector3i;
typedef class math::Vector3<uint> Vector3ui;
typedef class math::Vector3<uchar> Vector3u;
typedef class math::Vector3<float> Vector3f;

typedef class math::Matrix2<float> Matrix2f;
}//namespace math
