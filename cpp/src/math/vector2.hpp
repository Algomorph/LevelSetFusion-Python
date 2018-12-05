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
#include "math_utils.hpp"
#include "vector_base.hpp"
#include "platform_independence.hpp"

namespace math {


//======================================================================================================================
// Vector class with math operators: +, -, *, /, +=, -=, /=, [], ==, !=, T*(), etc. (inspired by InfiniTAM ORUtils)
//======================================================================================================================
template<class T>
class Vector2 : public Vector2_<T> {
public:
	typedef T value_type;
	static const int size = 2;
	_CPU_AND_GPU_CODE_ inline int get_size() const { return 2; }

	//===========================================================
	//                      Constructors
	//===========================================================
	_CPU_AND_GPU_CODE_ Vector2() = default; // Default constructor
	_CPU_AND_GPU_CODE_ explicit Vector2(const T& t) {
		this->x = t;
		this->y = t;
	} // Scalar constructor
	//TODO: ambiguity with above on calls like Vector2(0), provide meaningful alternative
//	_CPU_AND_GPU_CODE_ explicit Vector2(const T tp[]) {
//		this->x = tp[0];
//		this->y = tp[1];
//	} // Construct from array
	_CPU_AND_GPU_CODE_ Vector2(const T v0, const T v1) {
		this->x = v0;
		this->y = v1;
	} // Construct from explicit values
	_CPU_AND_GPU_CODE_ explicit Vector2(const Vector2_<T>& v) {
		this->x = v.x;
		this->y = v.y;
	}// copy constructor

	_CPU_AND_GPU_CODE_ explicit Vector2(const Vector3_<T>& u) {
		this->x = u.x;
		this->y = u.y;
	}

	_CPU_AND_GPU_CODE_ explicit Vector2(const Vector4_<T>& u) {
		this->x = u.x;
		this->y = u.y;
	}

	_CPU_AND_GPU_CODE_ inline Vector2<int> to_int() const {
		return {(int) ROUND(this->x), (int) ROUND(this->y)};
	}

	_CPU_AND_GPU_CODE_ inline Vector2<int> to_int_floor() const {
		return {(int) floor(this->x), (int) floor(this->y)};
	}

	_CPU_AND_GPU_CODE_ inline Vector2<unsigned char> to_uchar() const {
		Vector2<int> vi = to_int();
		return {(unsigned char) CLAMP(vi.x, 0, 255), (unsigned char) CLAMP(vi.y, 0, 255)};
	}

	_CPU_AND_GPU_CODE_ inline Vector2<float> toFloat() const {
		return {(float) this->x, (float) this->y};
	}

	_CPU_AND_GPU_CODE_ const T* getValues() const { return this->values; }

	_CPU_AND_GPU_CODE_ Vector2<T>& setValues(const T* rhs) {
		this->x = rhs[0];
		this->y = rhs[1];
		return *this;
	}

	bool is_zero() const{
		return this->x == 0.0f && this->y == 0.0f;
	}

	bool is_close_to_zero(T tolerance = 1e-10){
		return std::abs<T>(this->x) < tolerance && std::abs<T>(this->y) < tolerance;
	}

	// indexing operators
	_CPU_AND_GPU_CODE_ T& operator[](int i) { return this->values[i]; }

	_CPU_AND_GPU_CODE_ const T& operator[](int i) const { return this->values[i]; }

	// type-cast operators
	_CPU_AND_GPU_CODE_ operator T*() { return this->values; }

	_CPU_AND_GPU_CODE_ operator const T*() const { return this->values; }

	//===========================================================
	//                    Math operators
	//===========================================================

	// scalar multiply assign
	_CPU_AND_GPU_CODE_ friend Vector2<T>& operator*=(Vector2<T>& lhs, T d) {
		lhs.x *= d;
		lhs.y *= d;
		return lhs;
	}

	// component-wise vector multiply assign
	_CPU_AND_GPU_CODE_ friend Vector2<T>& operator*=(Vector2<T>& lhs, const Vector2<T>& rhs) {
		lhs.x *= rhs.x;
		lhs.y *= rhs.y;
		return lhs;
	}

	// scalar divide assign
	_CPU_AND_GPU_CODE_ friend Vector2<T>& operator/=(Vector2<T>& lhs, T d) {
		if (d == 0) return lhs;
		lhs.x /= d;
		lhs.y /= d;
		return lhs;
	}

	// component-wise vector divide assign
	_CPU_AND_GPU_CODE_ friend Vector2<T>& operator/=(Vector2<T>& lhs, const Vector2<T>& rhs) {
		lhs.x /= rhs.x;
		lhs.y /= rhs.y;
		return lhs;
	}

	// component-wise vector add assign
	_CPU_AND_GPU_CODE_ friend Vector2<T>& operator+=(Vector2<T>& lhs, const Vector2<T>& rhs) {
		lhs.x += rhs.x;
		lhs.y += rhs.y;
		return lhs;
	}

	// component-wise vector subtract assign
	_CPU_AND_GPU_CODE_ friend Vector2<T>& operator-=(Vector2<T>& lhs, const Vector2<T>& rhs) {
		lhs.x -= rhs.x;
		lhs.y -= rhs.y;
		return lhs;
	}

	// unary negate
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator-(const Vector2<T>& rhs) {
		Vector2<T> rv;
		rv.x = -rhs.x;
		rv.y = -rhs.y;
		return rv;
	}

	// vector add
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator+(const Vector2<T>& lhs, const Vector2<T>& rhs) {
		Vector2<T> rv(lhs);
		return rv += rhs;
	}

	// vector subtract
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator-(const Vector2<T>& lhs, const Vector2<T>& rhs) {
		Vector2<T> rv(lhs);
		return rv -= rhs;
	}

	// scalar multiply
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator*(const Vector2<T>& lhs, T rhs) {
		Vector2<T> rv(lhs);
		return rv *= rhs;
	}

	// scalar multiply
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator*(T lhs, const Vector2<T>& rhs) {
		Vector2<T> rv(lhs);
		return rv *= rhs;
	}

	// vector component-wise multiply
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator*(const Vector2<T>& lhs, const Vector2<T>& rhs) {
		Vector2<T> rv(lhs);
		return rv *= rhs;
	}

	// scalar multiply
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator/(const Vector2<T>& lhs, T rhs) {
		Vector2<T> rv(lhs);
		return rv /= rhs;
	}

	// vector component-wise multiply
	_CPU_AND_GPU_CODE_ friend Vector2<T> operator/(const Vector2<T>& lhs, const Vector2<T>& rhs) {
		Vector2<T> rv(lhs);
		return rv /= rhs;
	}

	//===========================================================
	//                   Comparison operators & functions
	//===========================================================

	// equality
	_CPU_AND_GPU_CODE_ friend bool operator==(const Vector2<T>& lhs, const Vector2<T>& rhs) {
		return (lhs.x == rhs.x) && (lhs.y == rhs.y);
	}

	// margin-comparison
	_CPU_AND_GPU_CODE_ bool is_close(const Vector2<T>& rhs, T margin=1e-10) {
		return std::abs<T>(this->x - rhs.x) < margin && std::abs<T>(this->y - rhs.y) < margin;
	}


	// inequality
	_CPU_AND_GPU_CODE_ friend bool operator!=(const Vector2<T>& lhs, const Vector2<T>& rhs) {
		return (lhs.x != rhs.x) || (lhs.y != rhs.y);
	}

	//===========================================================
	//                   Printing
	//===========================================================
	friend std::ostream& operator<<(std::ostream& os, const Vector2<T>& dt) {
		os << dt.x << ", " << dt.y;
		return os;
	}
};


} // namespace math
