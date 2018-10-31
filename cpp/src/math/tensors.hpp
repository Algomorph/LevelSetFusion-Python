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

//stdlib
#include <cstdlib>

//local
#include "vector2.hpp"
#include "matrix2.hpp"

//libraries
#include <Eigen/Eigen>
#include <iostream>

namespace math {
typedef Eigen::Matrix<math::Vector2<float>, Eigen::Dynamic, Eigen::Dynamic> MatrixXv2f;
typedef Eigen::Matrix<math::Matrix2<float>, Eigen::Dynamic, Eigen::Dynamic> MatrixXm2f;


template<typename TMatrix>
bool almost_equal(TMatrix matrix_a, TMatrix matrix_b, double tolerance = 1e-10) {
	if (matrix_a.rows() != matrix_b.rows() || matrix_a.cols() != matrix_b.rows()) {
		return false;
	}
	for (Eigen::Index index = 0; index < matrix_a.size(); index++) {
		if (!matrix_a(index).is_close(matrix_b(index), tolerance)) {
			return false;
		}
	}
	return true;
}

template<typename TMatrix>
bool almost_equal_verbose(TMatrix matrix_a, TMatrix matrix_b, double tolerance = 1e-10) {
	if (matrix_a.rows() != matrix_b.rows() || matrix_a.cols() != matrix_b.rows()) {
		std::cout << "Matrix dimensions don't match. Matrix a: " << matrix_a.cols() << " columns by " << matrix_a.rows()
		          << " rows, Matrix b: " << matrix_b.cols() << " columns by " << matrix_b.rows() << " rows."
		          << std::endl;
		return false;
	}
	for (Eigen::Index index = 0; index < matrix_a.size(); index++) {
		if (!matrix_a(index).is_close(matrix_b(index), tolerance)) {
			ldiv_t division_result = div(index, matrix_a.cols());
			long x = division_result.quot;
			long y = division_result.rem;
			std::cout << "Matrix entries are not within tolerance threshold of each other. First mismatch at row " << y
			          << ", column " << x << ". " << "Values: " << matrix_a(index) << " vs. " << matrix_b(index)
			          << ", difference: " << matrix_a(index) - matrix_b(index) << std::endl;
			return false;
		}
	}
	return true;
}

MatrixXv2f stack_as_xv2f(const Eigen::MatrixXf& matrix_a, const Eigen::MatrixXf& matrix_b);

}//namespace math


namespace Eigen {

template<>
struct NumTraits<math::Vector2<float>>
		: NumTraits<float> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector2<float> Real;
	typedef math::Vector2<float> NonInteger;
	typedef math::Vector2<float> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 2,
		MulCost = 6
	};
};

template<>
struct NumTraits<math::Vector2<double>>
		: NumTraits<double> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector2<double> Real;
	typedef math::Vector2<double> NonInteger;
	typedef math::Vector2<double> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 0,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 2,
		MulCost = 6
	};
};

template<>
struct NumTraits<math::Vector2<int>>
		: NumTraits<int> // permits to get the epsilon, dummy_precision, lowest, highest functions
{
	typedef math::Vector2<int> Real;
	typedef math::Vector2<int> Integer;
	typedef math::Vector2<int> Nested;
	enum {
		IsComplex = 0,
		IsInteger = 1,
		IsSigned = 1,
		RequireInitialization = 1,
		ReadCost = 1,
		AddCost = 2,
		MulCost = 6
	};
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<double>, float, BinaryOp> {
	typedef math::Vector2<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<double>, double, BinaryOp> {
	typedef math::Vector2<double> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<float>, float, BinaryOp> {
	typedef math::Vector2<float> ReturnType;
};

template<typename BinaryOp>
struct ScalarBinaryOpTraits<math::Vector2<float>, double, BinaryOp> {
	typedef math::Vector2<float> ReturnType;
};

}// namespacd Eigen

