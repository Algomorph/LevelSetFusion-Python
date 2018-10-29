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
//local
#include "data_term.hpp"
#include "boolean_operations.hpp"
#include "../math/typedefs.hpp"

namespace data_term {


/***
 * \brief Computes the local gradient of the data energy term for KillingFusion/SobolevFusion-based optimization on a
 * 2D grid at the specified location
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
                                      int i_col,
                                      int i_row,
                                      const eig::MatrixXf& live_gradient_x_field,
                                      const eig::MatrixXf& live_gradient_y_field,
                                      float& data_gradient_x, float& data_gradient_y,
                                      float& local_energy_contribution) {
	float live_sdf = warped_live_field(i_col, i_row);
	float canonical_sdf = canonical_field(i_col, i_row);
	float difference = live_sdf - canonical_sdf;
	float scaling_factor = 10.0F;
	float gradient_x = live_gradient_x_field(i_col, i_row);
	float gradient_y = live_gradient_y_field(i_col, i_row);


	data_gradient_x = difference * gradient_x * scaling_factor;
	data_gradient_y = difference * gradient_y * scaling_factor;
	local_energy_contribution = 0.5F * difference * difference;
}

template<bool TSkipTruncated>
inline void compute_data_term_gradient_aux(
		math::MatrixXv2f& data_term_gradient, float& data_term_energy,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		const math::MatrixXv2f& warped_live_field_gradient, float scaling_factor) {
	const eig::Index matrix_size = warped_live_field.size();
	const eig::Index column_count = warped_live_field.cols();
	const eig::Index row_count = warped_live_field.rows();

	data_term_gradient = math::MatrixXv2f(row_count, column_count);
#pragma omp parallel for
	for (eig::Index i_element = 0; i_element < matrix_size; i_element++) {
		// Any MatrixXf in Eigen is column-major
		// i_element = x * column_count + y
		ldiv_t division_result = div(i_element, column_count);
		int y = division_result.rem;
		int x = division_result.quot;
		float live_tsdf_value = warped_live_field(i_element);
		float canonical_tsdf_value = canonical_field(i_element);
		if (TSkipTruncated) {
			if (boolean_ops::is_outside_narrow_band(live_tsdf_value, canonical_tsdf_value)) {
				data_term_gradient(i_element) = math::Vector2f(0.0f);
				continue;
			}
		}
		float diff = live_tsdf_value - canonical_tsdf_value;
		data_term_gradient(i_element) = scaling_factor * diff * warped_live_field_gradient(i_element);
	}
}

/**
 * \brief Computes the gradient of the data energy term for KillingFusion/SobolevFusion-based optimization on a 2D grid
 * \details Goes over every location, regardless of whether the TSDF values are within the narrow band (are not truncated)
 * See Section 4.1 in KillingFusion[1] / 1.1 in KillingFusion Supplementary Material / 4.1 in SobolevFusion[2]
 * [1] M. Slavcheva, M. Baust, D. Cremers, and S. Ilic, “KillingFusion: Non-rigid 3D Reconstruction without Correspondences,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, no. 4, pp. 1386–1395.
 * [2] M. Slavcheva, M. Baust, and S. Ilic, “SobolevFusion : 3D Reconstruction of Scenes Undergoing Free Non-rigid Motion,” in Computer Vision and
 *
 * \param warped_live_field
 * \param canonical_field
 * \param warped_live_field_gradient
 * \param data_term_gradient
 * \param data_term_energy
 * \param scaling_factor
 */
void compute_data_term_gradient(
		math::MatrixXv2f& data_term_gradient, float& data_term_energy,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		const math::MatrixXv2f& warped_live_field_gradient, float scaling_factor) {
	compute_data_term_gradient_aux<false>(data_term_gradient, data_term_energy, warped_live_field, canonical_field,
	                                      warped_live_field_gradient, scaling_factor);
}

/**
 * \brief Does the same thing as compute_data_term_gradient(), except returns a zero vector as gradient for any locations
 * where both the passed warped live field and canonical field values are truncated
 * \param warped_live_field
 * \param canonical_field
 * \param warped_live_field_gradient
 * \param[out] data_term_gradient
 * \param[out] data_term_energy
 * \param[in] scaling_factor -- factor to scale the gradient. Usually, narrow-band half-width divided by the voxel size.
 */
void compute_data_term_gradient_within_band_union(
		math::MatrixXv2f& data_term_gradient, float& data_term_energy,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		const math::MatrixXv2f& warped_live_field_gradient, float scaling_factor) {
	compute_data_term_gradient_aux<true>(data_term_gradient, data_term_energy, warped_live_field, canonical_field,
	                                     warped_live_field_gradient, scaling_factor);
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