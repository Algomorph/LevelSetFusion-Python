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

namespace math{

//======================================================================================================================
//		                    Base Vector Structure (inspired by InfiniTAM ORUtils)
//======================================================================================================================

template <class T> struct Vector2_{
	union {
		struct { T x, y; }; // standard names for components
		struct { T s, t; }; // standard names for components
		struct { T u, v; }; // standard names for components
		struct { T width, height; }; // standard names for components
		T values[2];     // array access
	};
};

template <class T> struct Vector3_{
	union {
		struct{ T x, y, z; }; // standard names for components
		struct{ T r, g, b; }; // standard names for components
		struct{ T s, t, p; }; // standard names for components
		struct{ T u, v, w; }; // standard names for components
		T values[3];
	};
};

template <class T> struct Vector4_ {
	union {
		struct { T fx, fy, cx, cy; }; // names for components of a projection matrix
		struct { T x, y, z, w; }; // standard names for components
		struct { T r, g, b, a; }; // standard names for components
		struct { T s, t, p, q; }; // standard names for components
		struct { T u_x, u_y, v_x, v_y; }; // names for gradients of a vector field with u and v fields
		T values[4];
	};
};

template <class T> struct Vector6_ {
	union {
		struct { T min_x, min_y, min_z, max_x, max_y, max_z; };// standard names for components
		T values[6];
	};
};

template <class T> struct Vector9_ {
	union {
		struct { T u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z;};// names for gradients of a vector field with u, v, and w fields
		T values[9];
	};
};

template<class T, int s> struct VectorX_
{
	T values[s];
};


}
