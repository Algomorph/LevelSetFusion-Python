//  ================================================================
//  Created by Gregory Kramida on 10/29/18.
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

//libraries
#include <Eigen/Eigen>

//local
#include "../math/typedefs.hpp"

namespace eig = Eigen;

namespace traversal {


template<typename TFunctorFieldToField, typename TInField, typename TOutField>
inline void
traverse_2d_field_output_field_multithreaded(TOutField& out_field, const TInField& in_field,
                                             TFunctorFieldToField& functor) {
	const eig::Index matrix_size = in_field.size();
	const eig::Index column_count = in_field.cols();
	const eig::Index row_count = in_field.rows();
	eigen_assert((row_count == out_field.rows() && column_count == out_field.cols()) &&
	             "Argument matrices do not have the same dimensions.");

	out_field = TOutField(row_count, column_count);
#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < matrix_size; i_element++) {
		out_field(i_element) = functor(in_field(i_element));
	}
}


template<typename TFunctor2FieldsToField, typename TInFieldA, typename TInFieldB, typename TOutField>
inline void traverse_dual_2d_field_output_field_singlethreaded(TOutField& out_field, const TInFieldA& in_field_a,
                                                               const TInFieldB& in_field_b,
                                                               TFunctor2FieldsToField& functor) {
	const eig::Index matrix_size = in_field_a.size();
	const eig::Index column_count = in_field_a.cols();
	const eig::Index row_count = in_field_a.rows();
	eigen_assert((row_count == out_field.rows() && column_count == out_field.cols() &&
	              (row_count == in_field_b.rows() && column_count == in_field_b.cols())) &&
	             "Argument matrices do not have the same dimensions.");

	out_field = TOutField(row_count, column_count);
	for (eig::Index i_element = 0; i_element < matrix_size; i_element++) {
		out_field(i_element) = functor(in_field_a(i_element), in_field_a(i_element));
	}
}

template<typename TFunctor2FieldsToField, typename TInFieldA, typename TInFieldB, typename TOutField>
inline void traverse_dual_2d_field_output_field_multithreaded(TOutField& out_field, const TInFieldA& in_field_a,
                                                              const TInFieldB& in_field_b,
                                                              TFunctor2FieldsToField& functor) {
	const eig::Index matrix_size = in_field_a.size();
	const eig::Index column_count = in_field_a.cols();
	const eig::Index row_count = in_field_a.rows();
	eigen_assert((row_count == out_field.rows() && column_count == out_field.cols() &&
	              (row_count == in_field_b.rows() && column_count == in_field_b.cols())) &&
	             "Argument matrices do not have the same dimensions.");

	out_field = TOutField(row_count, column_count);
#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < matrix_size; i_element++) {
		out_field(i_element) = functor(in_field_a(i_element), in_field_a(i_element));
	}
}

template<typename TFunctor3FieldsToField, typename TInFieldA, typename TInFieldB, typename TInFieldC, typename TOutField>
inline void traverse_triple_2d_field_output_field_singlethreaded(TOutField& out_field, const TInFieldA& in_field_a,
                                                                const TInFieldB& in_field_b,
                                                                const TInFieldC& in_field_c,
                                                                TFunctor3FieldsToField& functor) {
	const eig::Index matrix_size = in_field_a.size();
	const eig::Index column_count = in_field_a.cols();
	const eig::Index row_count = in_field_a.rows();
	eigen_assert((row_count == out_field.rows() && column_count == out_field.cols() &&
	              (row_count == in_field_b.rows() && column_count == in_field_b.cols())) &&
	             "Argument matrices do not have the same dimensions.");

	out_field = TOutField(row_count, column_count);
	for (eig::Index i_element = 0; i_element < matrix_size; i_element++) {
		out_field(i_element) = functor(in_field_a(i_element), in_field_b(i_element), in_field_c(i_element));
	}
}

template<typename TFunctor3FieldsToField, typename TInFieldA, typename TInFieldB, typename TInFieldC, typename TOutField>
inline void traverse_triple_2d_field_output_field_multithreaded(TOutField& out_field, const TInFieldA& in_field_a,
                                                                const TInFieldB& in_field_b,
                                                                const TInFieldC& in_field_c,
                                                                TFunctor3FieldsToField& functor) {
	const eig::Index matrix_size = in_field_a.size();
	const eig::Index column_count = in_field_a.cols();
	const eig::Index row_count = in_field_a.rows();
	eigen_assert((row_count == out_field.rows() && column_count == out_field.cols() &&
	              (row_count == in_field_b.rows() && column_count == in_field_b.cols())) &&
	             "Argument matrices do not have the same dimensions.");

	out_field = TOutField(row_count, column_count);
#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < matrix_size; i_element++) {
		out_field(i_element) = functor(in_field_a(i_element), in_field_b(i_element), in_field_c(i_element));
	}
}

template<typename TFunctor3FieldsToField, typename TInFieldA, typename TInFieldB, typename TInFieldC, typename TOutField>
inline void traverse_triple_2d_field_using_coordinates_output_field_multithreaded(
		TOutField& out_field, const TInFieldA& in_field_a,
		const TInFieldB& in_field_b,
		const TInFieldC& in_field_c,
		TFunctor3FieldsToField& functor) {
	const int matrix_size = static_cast<int>(in_field_a.size());
	const int column_count = static_cast<int>(in_field_a.cols());
	const int row_count = static_cast<int>(in_field_a.rows());
	eigen_assert((row_count == out_field.rows() && column_count == out_field.cols() &&
	              (row_count == in_field_b.rows() && column_count == in_field_b.cols()) &&
	              (row_count == in_field_c.rows() && column_count == in_field_c.cols())) &&
	             "Argument matrices do not have the same dimensions.");

	out_field = TOutField(row_count, column_count);

#pragma omp parallel for
	for (int i_element = 0; i_element < matrix_size; i_element++) {
		div_t division_result = div(i_element, column_count);
		math::Vector2i coordinates(division_result.quot, division_result.rem);
		out_field(i_element) = functor(in_field_a(i_element), in_field_b(i_element), in_field_c(i_element),
		                               coordinates);
	}
}


}//namespace traversal