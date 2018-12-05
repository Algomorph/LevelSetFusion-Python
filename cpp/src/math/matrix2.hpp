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

//local
#include "matrix_base.hpp"
#include "vector2.hpp"
#include "vector3.hpp"
#include "platform_independence.hpp"


namespace math {

//======================================================================================================================
//                                      Matrix class with math operators
//======================================================================================================================
template<class T>
class Matrix2 : public Matrix2_<T> {
public:
	static const int size = 4;

	_CPU_AND_GPU_CODE_ Matrix2() {}

	_CPU_AND_GPU_CODE_ Matrix2(T t) { set_values(t); }

	_CPU_AND_GPU_CODE_ Matrix2(const T* m) { set_values(m); }

	/**
	 * \brief Constructor, fill in using xy, i.e. row-major, order!
	 * \param a00
	 * \param a01
	 * \param a10
	 * \param a11
	 */
	_CPU_AND_GPU_CODE_ Matrix2(T a00, T a01, T a10, T a11) {
		this->xy00 = a00;
		this->xy01 = a01;
		this->xy10 = a10;
		this->xy11 = a11;
	}

	_CPU_AND_GPU_CODE_ inline void get_values(T* mp) const { memcpy(mp, this->values, sizeof(T) * 4); }

	_CPU_AND_GPU_CODE_ inline const T* get_values() const { return this->values; }

	_CPU_AND_GPU_CODE_ inline Vector2<T> get_scale() const { return Vector3<T>(this->m00, this->m11); }

	// indexing
	_CPU_AND_GPU_CODE_ inline T& operator()(int x, int y) { return at(x, y); }

	_CPU_AND_GPU_CODE_ inline const T& operator()(int x, int y) const { return at(x, y); }

	_CPU_AND_GPU_CODE_ inline T& operator()(Vector2<int> pnt) { return at(pnt.x, pnt.y); }

	_CPU_AND_GPU_CODE_ inline const T& operator()(Vector2<int> pnt) const { return at(pnt.x, pnt.y); }

	_CPU_AND_GPU_CODE_ inline T& at(int x, int y) { return this->values[y | (x << 1)]; }

	_CPU_AND_GPU_CODE_ inline const T& at(int x, int y) const { return this->values[y | (x << 1)]; }

	// set values
	_CPU_AND_GPU_CODE_ inline void set_values(const T* mp) { memcpy(this->values, mp, sizeof(T) * 4); }

	_CPU_AND_GPU_CODE_ inline void set_values(T value) { for (int i = 0; i < 4; i++) this->values[i] = value; }

	_CPU_AND_GPU_CODE_ inline void set_to_zeros() { memset(this->values, 0, sizeof(T) * 4); }

	_CPU_AND_GPU_CODE_ inline void set_to_identity() {
		set_to_zeros();
		this->m00 = this->m11 = 1;
	}

	_CPU_AND_GPU_CODE_ inline void set_scale(T s) { this->m00 = this->m11 = s; }

	_CPU_AND_GPU_CODE_ inline void set_scale(const Vector2_<T>& s) {
		this->m00 = s[0];
		this->m11 = s[1];
	}

	// set slices
	_CPU_AND_GPU_CODE_ inline void set_row(int r, const Vector2_<T>& t) {
		for (int x = 0; x < 2; x++) at(x, r) = t[x];
	}

	_CPU_AND_GPU_CODE_ inline void set_column(int c, const Vector2_<T>& t) {
		memcpy(this->values + 2 * c, t.values, sizeof(T) * 2);
	}

	// get slices
	_CPU_AND_GPU_CODE_ inline Vector2<T> get_row(int r) const {
		Vector2<T> v;
		for (int x = 0; x < 2; x++) v[x] = at(x, r);
		return v;
	}

	_CPU_AND_GPU_CODE_ inline Vector2<T> get_column(int c) const {
		Vector2<T> v;
		memcpy(v.values, this->values + 2 * c, sizeof(T) * 2);
		return v;
	}

	// transpose
	_CPU_AND_GPU_CODE_ inline Matrix2 t() {
		Matrix2 transposed;
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				transposed(x, y) = at(y, x);
			}
		}
		return transposed;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator*(const Matrix2& lhs, const T& rhs) {
		Matrix2 r;
		for (int i = 0; i < 4; i++) {
			r.values[i] = lhs.values[i] * rhs;
		}
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator/(const Matrix2& lhs, const T& rhs) {
		Matrix2 r;
		for (int i = 0; i < 4; i++) {
			r.values[i] = lhs.values[i] / rhs;
		}
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator*(const Matrix2& lhs, const Matrix2& rhs) {
		Matrix2 r;
		r.set_to_zeros();
		for (int x = 0; x < 2; x++) {
			for (int y = 0; y < 2; y++) {
				for (int k = 0; k < 2; k++) {
					r(x, y) += lhs(k, y) * rhs(x, k);
				}
			}
		}
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator+(const Matrix2& lhs, const Matrix2& rhs) {
		Matrix2 res(lhs.values);
		return res += rhs;
	}

	_CPU_AND_GPU_CODE_ inline Vector2<T> operator*(const Vector2<T>& rhs) const {
		Vector2<T> r;
		r[0] = this->values[0] * rhs[0] + this->values[2] * rhs[1];
		r[1] = this->values[1] * rhs[0] + this->values[3] * rhs[1];
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Vector2<T> operator*(const Vector2<T>& lhs, const Matrix2& rhs) {
		Vector2<T> r;
		for (int x = 0; x < 4; x++) {
			r[x] = lhs[0] * rhs(x, 0) + lhs[1] * rhs(x, 1) + lhs[2] * rhs(x, 2) + lhs[3] * rhs(x, 3);
		}
		return r;
	}

	_CPU_AND_GPU_CODE_ inline Matrix2& operator+=(const T& r) {
		for (int i = 0; i < 4; ++i) {
			this->values[i] += r;
		}
		return *this;
	}

	_CPU_AND_GPU_CODE_ inline Matrix2& operator-=(const T& r) {
		for (int i = 0; i < 4; ++i) {
			this->values[i] -= r;
		}
		return *this;
	}

	_CPU_AND_GPU_CODE_ inline Matrix2& operator*=(const T& r) {
		for (int i = 0; i < 4; ++i) {
			this->values[i] *= r;
		}
		return *this;
	}

	_CPU_AND_GPU_CODE_ inline Matrix2& operator/=(const T& r) {
		for (int i = 0; i < 4; ++i) {
			this->values[i] /= r;
		}
		return *this;
	}

	_CPU_AND_GPU_CODE_ inline Matrix2& operator+=(const Matrix2& mat) {
		for (int i = 0; i < 4; ++i) {
			this->values[i] += mat.values[i];
		}
		return *this;
	}

	_CPU_AND_GPU_CODE_ inline Matrix2& operator-=(const Matrix2& mat) {
		for (int i = 0; i < 4; ++i) {
			this->values[i] -= mat.values[i];
		}
		return *this;
	}

	//=====================================================================================
	//region              Comparison Operators and Functions
	//=====================================================================================

/*
 TODO: (following three functions) test whether returning false directly on first comparison failure significantly
 improves CPU performance or affects GPU performance
*/
	_CPU_AND_GPU_CODE_ inline friend bool operator==(const Matrix2& lhs, const Matrix2& rhs) {
		bool r = lhs.values[0] == rhs.values[0];
		for (int i = 1; i < 4; i++) {
			r &= lhs.values[i] == rhs.values[i];
		}
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend bool operator!=(const Matrix2& lhs, const Matrix2& rhs) {
		bool r = lhs.values[0] != rhs.values[0];
		for (int i = 1; i < 4; i++) {
			r |= lhs.values[i] != rhs.values[i];
		}
		return r;
	}

	// margin-comparison (for floating-point T types) //TODO: use consistent naming,  i.e. tensor.h uses "is_approx"
	_CPU_AND_GPU_CODE_ bool is_close(const Matrix2<T>& rhs, T margin = 1e-10) {
		bool r = std::abs<T>(this->values[0] - rhs.values[0]) < margin;
		for (int i = 1; i < 4; i++) {
			r &= std::abs<T>(this->values[i] - rhs.values[i]) < margin;
		}
		return r;
	}
	//endregion ============================================================================

	// Matrix determinant
	_CPU_AND_GPU_CODE_ inline T det() const {
		return this->xy00 * this->xy11 - this->xy01 * this->xy10;
	}

	// The inverse matrix for float/double type
	_CPU_AND_GPU_CODE_ inline bool inv(Matrix2& out) const {
		T determinant = this->det();
		if (determinant == 0) {
			out.set_to_zeros();
		}
		out.xy00 = this->xy11 / determinant;
		out.xy10 = -this->xy10 / determinant;
		out.xy01 = -this->xy01 / determinant;
		out.xy11 = this->xy00 / determinant;
		return true;
	}

	/**
	 * \brief CPU-only readable output function, the matrix is printed in row-major order
	 * \details printed on a single line to be readable when Matrix2<T> is used within an Eigen::Matrix and the whole
	 * outer matrix is printed
	 */
	friend std::ostream& operator<<(std::ostream& os, const Matrix2<T>& dt) {
		os << dt(0, 0) << ", " << dt(0, 1) << " \\ " << dt(1, 0) << ", " << dt(1, 1);
		return os;
	}
};

}//namespace math

namespace Eigen{

template<typename T>
	struct NumTraits<math::Matrix2<T>>
		: NumTraits<T> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Matrix2<T> Real;
	typedef math::Matrix2<T> NonInteger;
	typedef math::Matrix2<T> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 4,
		AddCost = 4,
		MulCost = 24
	};
};

}//namespace Eigen