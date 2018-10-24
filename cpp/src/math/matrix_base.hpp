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

//======================================================================================================================
//						Basic Matrix Structure (inspired by InfiniTAM ORUtils)
//======================================================================================================================

template <class T> struct Matrix2_{
	union {
		struct { // Warning: see the header in this file for the special matrix order
			T yx00, yx01;	// |0, 2|
			T yx10, yx11;	// |1, 3|
		};
		struct { // Warning: see the header in this file for the special matrix order
			T xy00, xy10;	// |0, 2|
			T xy01, xy11;	// |1, 3|
		};
		T m[4];
	};
};

template <class T> struct Matrix3_{
	union { // Warning: see the header in this file for the special matrix order
		struct { //ordered struct representation
			T yx00, yx01, yx02; // |0, 3, 6|
			T yx10, yx11, yx12; // |1, 4, 7|
			T yx20, yx21, yx22; // |2, 5, 8|
		};
		struct { // Warning: see the header in this file for the special matrix order
			T xy00, xy10, xy20;	// |0, 3, 6|
			T xy01, xy11, xy21;	// |1, 4, 7|
			T xy02, xy12, xy22;	// |2, 5, 8|
		};
		struct { //Jacobian of [u v w] struct representation
			T u_x, v_x, w_x; // |0, 3, 6|
			T u_y, v_y, w_y; // |1, 4, 7|
			T u_z, v_z, w_z; // |2, 5, 8|
		};
		struct { //Hermitian struct representation
			T xx, xy, xz; // |0, 3, 6|
			T yx, yy, yz; // |1, 4, 7|
			T zx, zy, zz; // |2, 5, 8|
		};
		T m[9];
	};
};

template <class T> struct Matrix4_{
	union {
		struct { // Warning: see the header in this file for the special matrix order
			T yx00, yx01, yx02, yx03;	// |0, 4, 8,  12|    |yx00, yx10, yx20, yx30|
			T yx10, yx11, yx12, yx13;	// |1, 5, 9,  13|    |yx01, yx11, yx21, yx31|
			T yx20, yx21, yx22, yx23;	// |2, 6, 10, 14|    |yx02, yx12, yx22, yx32|
			T yx30, yx31, yx32, yx33;	// |3, 7, 11, 15|    |yx03, yx13, yx23, yx33|
		};
		struct { // Warning: see the header in this file for the special matrix order
			T xy00, xy10, xy20, xy30;	// |0, 4, 8,  12|
			T xy01, xy11, xy21, xy31;	// |1, 5, 9,  13|
			T xy02, xy12, xy22, xy32;	// |2, 6, 10, 14|
			T xy03, xy13, xy23, xy33;	// |3, 7, 11, 15|
		};
		T m[16];
	};
};

template<class T, int s> struct MatrixSQX_{
	int dim;
	int sq;
	T m[s*s];
};
}
