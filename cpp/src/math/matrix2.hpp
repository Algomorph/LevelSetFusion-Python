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
#include "vector_base.hpp"
#include "platform_independence.hpp"


namespace math {

template<class T>
class Vector2;
template<class T>
class Vector3;

//////////////////////////////////////////////////////////////////////////
// Matrix class with math operators
//////////////////////////////////////////////////////////////////////////
template<class T>
class Matrix2 : public Matrix2_ < T >
{
public:
	_CPU_AND_GPU_CODE_ Matrix2() {}
	_CPU_AND_GPU_CODE_ Matrix2(T t) { setValues(t); }
	_CPU_AND_GPU_CODE_ Matrix2(const T *m)	{ setValues(m); }
	_CPU_AND_GPU_CODE_ Matrix2(T a00, T a01, T a02, T a03, T a10, T a11, T a12, T a13, T a20, T a21, T a22, T a23, T a30, T a31, T a32, T a33)	{
		this->m00 = a00; this->m01 = a01; this->m02 = a02; this->m03 = a03;
		this->m10 = a10; this->m11 = a11; this->m12 = a12; this->m13 = a13;
		this->m20 = a20; this->m21 = a21; this->m22 = a22; this->m23 = a23;
		this->m30 = a30; this->m31 = a31; this->m32 = a32; this->m33 = a33;
	}

	_CPU_AND_GPU_CODE_ inline void getValues(T *mp) const	{ memcpy(mp, this->m, sizeof(T) * 16); }
	_CPU_AND_GPU_CODE_ inline const T *getValues() const { return this->m; }
	_CPU_AND_GPU_CODE_ inline Vector3<T> getScale() const { return Vector3<T>(this->m00, this->m11, this->m22); }

	// Element access
	_CPU_AND_GPU_CODE_ inline T &operator()(int x, int y)	{ return at(x, y); }
	_CPU_AND_GPU_CODE_ inline const T &operator()(int x, int y) const	{ return at(x, y); }
	_CPU_AND_GPU_CODE_ inline T &operator()(Vector2<int> pnt)	{ return at(pnt.x, pnt.y); }
	_CPU_AND_GPU_CODE_ inline const T &operator()(Vector2<int> pnt) const	{ return at(pnt.x, pnt.y); }
	_CPU_AND_GPU_CODE_ inline T &at(int x, int y) { return this->m[y | (x << 2)]; }
	_CPU_AND_GPU_CODE_ inline const T &at(int x, int y) const { return this->m[y | (x << 2)]; }

	// set values
	_CPU_AND_GPU_CODE_ inline void setValues(const T *mp) { memcpy(this->m, mp, sizeof(T) * 16); }
	_CPU_AND_GPU_CODE_ inline void setValues(T r)	{ for (int i = 0; i < 16; i++)	this->m[i] = r; }
	_CPU_AND_GPU_CODE_ inline void setZeros() { memset(this->m, 0, sizeof(T) * 16); }
	_CPU_AND_GPU_CODE_ inline void setIdentity() { setZeros(); this->m00 = this->m11 = this->m22 = this->m33 = 1; }
	_CPU_AND_GPU_CODE_ inline void setScale(T s) { this->m00 = this->m11 = this->m22 = s; }
	_CPU_AND_GPU_CODE_ inline void setScale(const Vector3_<T> &s) { this->m00 = s[0]; this->m11 = s[1]; this->m22 = s[2]; }
	_CPU_AND_GPU_CODE_ inline void setTranslate(const Vector3_<T> &t) { for (int y = 0; y < 3; y++) at(3, y) = t[y]; }
	_CPU_AND_GPU_CODE_ inline void setRow(int r, const Vector4_<T> &t){ for (int x = 0; x < 4; x++) at(x, r) = t[x]; }
	_CPU_AND_GPU_CODE_ inline void setColumn(int c, const Vector4_<T> &t) { memcpy(this->m + 4 * c, t.values, sizeof(T) * 4); }

	// get values
	_CPU_AND_GPU_CODE_ inline Vector4<T> getRow(int r) const { Vector4<T> v; for (int x = 0; x < 4; x++) v[x] = at(x, r); return v; }
	_CPU_AND_GPU_CODE_ inline Vector4<T> getColumn(int c) const { Vector4<T> v; memcpy(v.values, this->m + 4 * c, sizeof(T) * 4); return v; }
	_CPU_AND_GPU_CODE_ inline Matrix2 t() { // transpose
		Matrix2 mtrans;
		for (int x = 0; x < 4; x++)	for (int y = 0; y < 4; y++)
				mtrans(x, y) = at(y, x);
		return mtrans;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator * (const Matrix2 &lhs, const T &rhs)	{
		Matrix2 r;
		for (int i = 0; i < 16; i++) r.m[i] = lhs.m[i] * rhs;
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator / (const Matrix2 &lhs, const T &rhs)	{
		Matrix2 r;
		for (int i = 0; i < 16; i++) r.m[i] = lhs.m[i] / rhs;
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator * (const Matrix2 &lhs, const Matrix2 &rhs)	{
		Matrix2 r;
		r.setZeros();
		for (int x = 0; x < 4; x++) for (int y = 0; y < 4; y++) for (int k = 0; k < 4; k++)
					r(x, y) += lhs(k, y) * rhs(x, k);
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Matrix2 operator + (const Matrix2 &lhs, const Matrix2 &rhs) {
		Matrix2 res(lhs.m);
		return res += rhs;
	}

	_CPU_AND_GPU_CODE_ inline Vector4<T> operator *(const Vector4<T> &rhs) const {
		Vector4<T> r;
		r[0] = this->m[0] * rhs[0] + this->m[4] * rhs[1] + this->m[8] * rhs[2] + this->m[12] * rhs[3];
		r[1] = this->m[1] * rhs[0] + this->m[5] * rhs[1] + this->m[9] * rhs[2] + this->m[13] * rhs[3];
		r[2] = this->m[2] * rhs[0] + this->m[6] * rhs[1] + this->m[10] * rhs[2] + this->m[14] * rhs[3];
		r[3] = this->m[3] * rhs[0] + this->m[7] * rhs[1] + this->m[11] * rhs[2] + this->m[15] * rhs[3];
		return r;
	}

	// Used as a projection matrix to multiply with the Vector3
	_CPU_AND_GPU_CODE_ inline Vector3<T> operator *(const Vector3<T> &rhs) const {
		Vector3<T> r;
		r[0] = this->m[0] * rhs[0] + this->m[4] * rhs[1] + this->m[8] * rhs[2] + this->m[12];
		r[1] = this->m[1] * rhs[0] + this->m[5] * rhs[1] + this->m[9] * rhs[2] + this->m[13];
		r[2] = this->m[2] * rhs[0] + this->m[6] * rhs[1] + this->m[10] * rhs[2] + this->m[14];
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend Vector4<T> operator *(const Vector4<T> &lhs, const Matrix2 &rhs){
		Vector4<T> r;
		for (int x = 0; x < 4; x++)
			r[x] = lhs[0] * rhs(x, 0) + lhs[1] * rhs(x, 1) + lhs[2] * rhs(x, 2) + lhs[3] * rhs(x, 3);
		return r;
	}

	_CPU_AND_GPU_CODE_ inline Matrix2& operator += (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] += r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix2& operator -= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] -= r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix2& operator *= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] *= r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix2& operator /= (const T &r) { for (int i = 0; i < 16; ++i) this->m[i] /= r; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix2 &operator += (const Matrix2 &mat) { for (int i = 0; i < 16; ++i) this->m[i] += mat.m[i]; return *this; }
	_CPU_AND_GPU_CODE_ inline Matrix2 &operator -= (const Matrix2 &mat) { for (int i = 0; i < 16; ++i) this->m[i] -= mat.m[i]; return *this; }

	_CPU_AND_GPU_CODE_ inline friend bool operator == (const Matrix2 &lhs, const Matrix2 &rhs) {
		bool r = lhs.m[0] == rhs.m[0];
		for (int i = 1; i < 16; i++)
			r &= lhs.m[i] == rhs.m[i];
		return r;
	}

	_CPU_AND_GPU_CODE_ inline friend bool operator != (const Matrix2 &lhs, const Matrix2 &rhs) {
		bool r = lhs.m[0] != rhs.m[0];
		for (int i = 1; i < 16; i++)
			r |= lhs.m[i] != rhs.m[i];
		return r;
	}

	// The inverse matrix for float/double type
	_CPU_AND_GPU_CODE_ inline bool inv(Matrix2 &out) const {
		T tmp[12], src[16], det;
		T *dst = out.m;
		for (int i = 0; i < 4; i++) {
			src[i] = this->m[i * 4];
			src[i + 4] = this->m[i * 4 + 1];
			src[i + 8] = this->m[i * 4 + 2];
			src[i + 12] = this->m[i * 4 + 3];
		}

		tmp[0] = src[10] * src[15];
		tmp[1] = src[11] * src[14];
		tmp[2] = src[9] * src[15];
		tmp[3] = src[11] * src[13];
		tmp[4] = src[9] * src[14];
		tmp[5] = src[10] * src[13];
		tmp[6] = src[8] * src[15];
		tmp[7] = src[11] * src[12];
		tmp[8] = src[8] * src[14];
		tmp[9] = src[10] * src[12];
		tmp[10] = src[8] * src[13];
		tmp[11] = src[9] * src[12];

		dst[0] = (tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7]) - (tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7]);
		dst[1] = (tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7]) - (tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7]);
		dst[2] = (tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7]) - (tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7]);
		dst[3] = (tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6]) - (tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6]);

		det = src[0] * dst[0] + src[1] * dst[1] + src[2] * dst[2] + src[3] * dst[3];
		if (det == 0.0f)
			return false;

		dst[4] = (tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3]) - (tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3]);
		dst[5] = (tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3]) - (tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3]);
		dst[6] = (tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3]) - (tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3]);
		dst[7] = (tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2]) - (tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2]);

		tmp[0] = src[2] * src[7];
		tmp[1] = src[3] * src[6];
		tmp[2] = src[1] * src[7];
		tmp[3] = src[3] * src[5];
		tmp[4] = src[1] * src[6];
		tmp[5] = src[2] * src[5];
		tmp[6] = src[0] * src[7];
		tmp[7] = src[3] * src[4];
		tmp[8] = src[0] * src[6];
		tmp[9] = src[2] * src[4];
		tmp[10] = src[0] * src[5];
		tmp[11] = src[1] * src[4];

		dst[8] = (tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15]) - (tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15]);
		dst[9] = (tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15]) - (tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15]);
		dst[10] = (tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15]) - (tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15]);
		dst[11] = (tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14]) - (tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14]);
		dst[12] = (tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9]) - (tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10]);
		dst[13] = (tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10]) - (tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8]);
		dst[14] = (tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8]) - (tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9]);
		dst[15] = (tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9]) - (tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8]);

		out *= 1 / det;
		return true;
	}

	friend std::ostream& operator<<(std::ostream& os, const Matrix2<T>& dt) {
		for (int y = 0; y < 4; y++)
			os << dt(0, y) << ", " << dt(1, y) << ", " << dt(2, y) << ", " << dt(3, y) << "\n";
		return os;
	}
};

}//namespace math