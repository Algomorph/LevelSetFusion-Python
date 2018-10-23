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

namespace math{
template <class T> class Vector2;
template <class T> class Vector3;
template <class T> class Vector4;
template <class T, int s> class VectorX;

//////////////////////////////////////////////////////////////////////////
//						Basic Matrix Structure
//////////////////////////////////////////////////////////////////////////

template <class T> struct Matrix4_{
	union {
		struct { // Warning: see the header in this file for the special matrix order
			T m00, m01, m02, m03;	// |0, 4, 8,  12|    |m00, m10, m20, m30|
			T m10, m11, m12, m13;	// |1, 5, 9,  13|    |m01, m11, m21, m31|
			T m20, m21, m22, m23;	// |2, 6, 10, 14|    |m02, m12, m22, m32|
			T m30, m31, m32, m33;	// |3, 7, 11, 15|    |m03, m13, m23, m33|
		};
		T m[16];
	};
};

template <class T> struct Matrix3_{
	union { // Warning: see the header in this file for the special matrix order
		struct { //ordered struct representation
			T m00, m01, m02; // |0, 3, 6|     |m00, m10, m20|
			T m10, m11, m12; // |1, 4, 7|     |m01, m11, m21|
			T m20, m21, m22; // |2, 5, 8|     |m02, m12, m22|
		};
		struct { //Jacobian struct representation
			T u_x, v_x, w_x; // |0, 3, 6|     |u_x, u_y, u_z|
			T u_y, v_y, w_y; // |1, 4, 7|     |v_x, v_y, v_z|
			T u_z, v_z, w_z; // |2, 5, 8|     |w_x, w_y, w_z|
		};
		struct { //Hermitian struct representation
			T xx, xy, xz; // |0, 3, 6|    |xx, yy, zx|
			T yx, yy, yz; // |1, 4, 7|    |xy, yy, zy|
			T zx, zy, zz; // |2, 5, 8|    |xz, yz, zz|
		};
		T m[9];
	};
};

template<class T, int s> struct MatrixSQX_{
	int dim;
	int sq;
	T m[s*s];
};
}
