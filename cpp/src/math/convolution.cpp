//  ================================================================
//  Created by Gregory Kramida on 11/3/18.
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

#include "convolution.hpp"
#include "typedefs.hpp"

namespace math {

inline
static math::Vector2f buffer_convolve_helper_preserve_zeros(math::Vector2f* buffer,
                                                            const eig::VectorXf& kernel_1d, int buffer_read_index,
                                                            int kernel_size,
                                                            const math::Vector2f& original_vector) {
	if (original_vector.is_zero()) {
		return {0, 0};
	}
	int i_kernel_value = 0;
	float x, y;
	x = y = 0.0f;
	for (int i_buffered_vector = buffer_read_index;
	     i_buffered_vector < kernel_size; i_buffered_vector++, i_kernel_value++) {
		float kernel_value = kernel_1d(i_kernel_value);
		math::Vector2f& buffered_vector = buffer[i_buffered_vector];
		x += buffered_vector.x * kernel_value;
		y += buffered_vector.y * kernel_value;
	}
	for (int i_buffered_vector = 0; i_buffered_vector < buffer_read_index; i_buffered_vector++, i_kernel_value++) {
		float kernel_value = kernel_1d(i_kernel_value);
		math::Vector2f& buffered_vector = buffer[i_buffered_vector];
		x += buffered_vector.x * kernel_value;
		y += buffered_vector.y * kernel_value;
	}
	return {x, y};
}

void convolve_with_kernel_preserve_zeros(MatrixXv2f& field, const eig::VectorXf& kernel_1d) {
	eig::Index row_count = field.rows();
	eig::Index column_count = field.cols();

	eig::VectorXf kernel_inverted(kernel_1d.size());

	//flip kernel, see def of discrete convolution on Wikipedia
	for (eig::Index i_kernel_element = 0; i_kernel_element < kernel_1d.size(); i_kernel_element++) {
		kernel_inverted(kernel_1d.size() - i_kernel_element - 1) = kernel_1d(i_kernel_element);
	}

	int kernel_size = static_cast<int>(kernel_inverted.size());
	int kernel_half_size = kernel_size / 2;


	math::Vector2f buffer[kernel_size];
	MatrixXv2f y_convolved = MatrixXv2f::Zero(field.rows(), field.cols());

//#pragma omp parallel for private(buffer)
	for (eig::Index i_col = 0; i_col < column_count; i_col++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_row_to_sample = 0;
		//fill buffer up to the last value
		for (; i_row_to_sample < kernel_half_size; i_row_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col);
		}
		int i_buffer_read_index = 0;
		eig::Index i_row_to_write = 0;
		for (; i_row_to_sample < row_count; i_row_to_write++, i_row_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = field(i_row_to_sample, i_col); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
					                                      kernel_size, field(i_row_to_write, i_col));
		}
		for (; i_row_to_write < row_count; i_row_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			y_convolved(i_row_to_write, i_col) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
					                                      kernel_size, field(i_row_to_write, i_col));
		}
	}
//#pragma omp parallel for private(buffer)
	for (eig::Index i_row = 0; i_row < row_count; i_row++) {
		int i_buffer_write_index = 0;
		for (; i_buffer_write_index < kernel_half_size + 1; i_buffer_write_index++) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f); // fill buffer with empty value
		}
		eig::Index i_col_to_sample = 0;
		//fill buffer up to the last value
		for (; i_col_to_sample < kernel_half_size; i_col_to_sample++, i_buffer_write_index++) {
			buffer[i_buffer_write_index] = y_convolved(i_row, i_col_to_sample);
		}
		int i_buffer_read_index = 0;
		eig::Index i_col_to_write = 0;
		for (; i_col_to_sample < column_count; i_col_to_write++, i_col_to_sample++,
				i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = field(i_row, i_col_to_sample); // fill buffer with next value
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			field(i_row, i_col_to_write) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
					                                      kernel_size, y_convolved(i_row, i_col_to_write));
		}
		for (; i_col_to_write < column_count; i_col_to_write++, i_buffer_write_index = i_buffer_read_index) {
			buffer[i_buffer_write_index] = math::Vector2f(0.0f);
			i_buffer_read_index = (i_buffer_write_index + 1) % kernel_size;
			field(i_row, i_col_to_write) =
					buffer_convolve_helper_preserve_zeros(buffer, kernel_inverted, i_buffer_read_index,
					                                      kernel_size, y_convolved(i_row, i_col_to_write));
		}
	}
}

}//namespace math
