//  ================================================================
//  Created by Gregory Kramida on 10/9/18.
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
#include "data_term.hpp"

namespace data_term {
/**
 * Compute numerical gradient of given matrix. Uses forward and backward differences at the boundary matrix entries,
 * central difference formula for all other entries.
 * Reference: https://en.wikipedia.org/wiki/Numerical_differentiation ;
 * Central difference formula: (f(x + h) - f(x-h))/2h, with h = 1 matrix grid cell
 *
 * @param[in] field scalar field representing the implicit function to differentiate
 * @param[out] live_gradient_x
 * @param[out] live_gradient_y
 */
void gradient(const eig::MatrixXf& field, eig::MatrixXf& live_gradient_x, eig::MatrixXf& live_gradient_y) {
    eigen_assert(field.rows() == live_gradient_x.rows() && field.cols() == live_gradient_x.cols() &&
                         field.rows() == live_gradient_y.rows() && field.cols() == live_gradient_y.cols());
#pragma omp parallel for
	for (eig::Index i_col = 0; i_col < field.cols(); i_col++){
		float prev_row_val = field(0,i_col);
		float row_val = field(1, i_col);
        live_gradient_y(0,i_col) = row_val - prev_row_val;
        eig::Index i_row;
		for(i_row = 1; i_row < field.rows()-1; i_row++){
			float next_row_val = field(i_row+1,i_col);
			live_gradient_y(i_row,i_col) = 0.5*(next_row_val - prev_row_val);
			prev_row_val = row_val;
			row_val = next_row_val;
		}
        live_gradient_y(i_row,i_col) = row_val - prev_row_val;
	}
#pragma omp parallel for
	for (eig::Index  i_row = 0; i_row < field.rows(); i_row++){
		float prev_col_val = field(i_row, 0);
		float col_val = field(i_row, 1);
        live_gradient_x(i_row,0) = col_val - prev_col_val;
        eig::Index  i_col;
		for(i_col = 1; i_col < field.cols()-1; i_col++){
			float next_col_val = field(i_row,i_col+1);
			live_gradient_x(i_row,i_col) = 0.5*(next_col_val - prev_col_val);
			prev_col_val = col_val;
			col_val = next_col_val;
		}
        live_gradient_x(i_row, i_col) = col_val - prev_col_val;
	}
}

/***
 * \brief Computes data term for KillingFusion/SobolevFusion-based optimization on a 2D grid at the specified location
 * \details See Section 4.1 in KillingFusion[1] / 1.1 in KillingFusion Supplementary Material / 4.1 in SobolevFusion[2]
 * [1] M. Slavcheva, M. Baust, D. Cremers, and S. Ilic, “KillingFusion: Non-rigid 3D Reconstruction without Correspondences,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, no. 4, pp. 1386–1395.
 * [2] M. Slavcheva, M. Baust, and S. Ilic, “SobolevFusion : 3D Reconstruction of Scenes Undergoing Free Non-rigid Motion,” in Computer Vision and Pattern Recognition, 2018.
 * \param warped_live_field warped version of the live SDF grid / field
 * \param canonical_field canonical SDF grid / field
 * \param x coordinate of the desired location
 * \param y coordinate of the desired location
 * \param live_gradient_x_field precomputed x gradient of warped_live_field
 * \param live_gradient_y_field precomputed y gradient of warped_live_field
 * \param[out] data_gradient_x  x, or u-component of the data term gradient
 * \param[out] data_gradient_y  y, or v-component of the data term gradient
 * \param[out] local_energy_contribution contribution to the data energy
 */
void compute_local_data_term_gradient(const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
                                      int x,
                                      int y,
                                      const eig::MatrixXf& live_gradient_x_field,
                                      const eig::MatrixXf& live_gradient_y_field,
                                      float& data_gradient_x, float& data_gradient_y,
                                      float& local_energy_contribution) {
	float live_sdf = warped_live_field(y, x);
	float canonical_sdf = canonical_field(y, x);
	float difference = live_sdf - canonical_sdf;
	float scaling_factor = 10.0F;
	float gradient_x = live_gradient_x_field(y, x);
	float gradient_y = live_gradient_y_field(y, x);


	data_gradient_x = difference * gradient_x * scaling_factor;
	data_gradient_y = difference * gradient_y * scaling_factor;
	local_energy_contribution = 0.5F * difference * difference;
}

bp::tuple py_data_term_at_location(eig::MatrixXf warped_live_field, eig::MatrixXf canonical_field, int x, int y,
                                   eig::MatrixXf live_gradient_x_field, eig::MatrixXf live_gradient_y_field) {

	float data_gradient_x, data_gradient_y, local_energy_contribution;
	compute_local_data_term_gradient(warped_live_field, canonical_field, x, y, live_gradient_x_field,
	                                 live_gradient_y_field,
	                                 data_gradient_x, data_gradient_y, local_energy_contribution);
	eig::RowVector2f data_gradient;
	data_gradient(0) = data_gradient_x;
	data_gradient(1) = data_gradient_y;
	bp::object data_gradient_out(data_gradient);
	return bp::make_tuple(data_gradient_out, local_energy_contribution);
}


}//namespace data_term