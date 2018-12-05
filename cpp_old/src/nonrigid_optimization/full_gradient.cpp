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

//local
#include "data_term.hpp"
#include "full_gradient.hpp"
#include "boolean_operations.hpp"
#include "../math/gradients.hpp"

namespace nonrigid_optimization {
/**
 * \brief Computes energy gradient KillingFusion[1]/SobolevFusion[2]-based optimization on a 2D grid
 * \details
 * [1] M. Slavcheva, M. Baust, D. Cremers, and S. Ilic, “KillingFusion: Non-rigid 3D Reconstruction without Correspondences,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, no. 4, pp. 1386–1395.
 * [2] M. Slavcheva, M. Baust, and S. Ilic, “SobolevFusion : 3D Reconstruction of Scenes Undergoing Free Non-rigid Motion,” in Computer Vision and Pattern Recognition, 2018.
 * @param[in] warped_live_field scalar field representing the implicit TSDF generated from a live (new) depth
 * image/frame and (potentially) warped
 * @param[in] canonical_field scalar field representing the TSDF representing the fused aggregate
 * @param[out] gradient_field_x a field with the x-component of the computed gradient, same dimensions as either of
 * the two input matrices, should not be initialized
 * @param[out] gradient_field_y a field with the y-component of the computed gradient, same dimensions as either of
 * the two input matrices, should not be initialized
 * @param[in] band_union_only when set to true, will return zeroes at entries corresponding to any locations which have
 * truncated values both in the warped live and the canonical scalar fields
 */
void compute_energy_gradient(const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
                             const eig::MatrixXf& warp_field_x, const eig::MatrixXf& warp_field_y,
                             eig::MatrixXf& gradient_field_x, eig::MatrixXf& gradient_field_y,
                             bool band_union_only) {


	eig::Index row_count = canonical_field.rows();
	eig::Index column_count = canonical_field.cols();

	// sanity check
	eigen_assert((row_count == warped_live_field.rows() && column_count == warped_live_field.cols())
	             && "Dimensions of canonical and live fields must match");

	// initialize both output fields to zeros
	gradient_field_x = eig::MatrixXf::Zero(row_count, column_count);
	gradient_field_y = eig::MatrixXf::Zero(row_count, column_count);


	eig::Index entry_count = canonical_field.size();

	eig::MatrixXf live_gradient_x_field, live_gradient_y_field;

	// compute warped live numerical gradient
	math::scalar_field_gradient(warped_live_field, live_gradient_x_field, live_gradient_y_field);

#pragma omp parallel for
	for (int i_element = 0; i_element < entry_count; i_element++) {
		float live_tsdf_value = warped_live_field(i_element);
		float canonical_tsdf_value = canonical_field(i_element);
		if (band_union_only && is_outside_narrow_band_tolerance(live_tsdf_value, canonical_tsdf_value)){
			continue;
		}

		// Any MatrixXf in Eigen is column-major
		// i_element = i_col * column_count + i_row
		div_t division_result = div(i_element, static_cast<int>(column_count));
		int i_col = division_result.quot;
		int i_row = division_result.rem;
		float local_data_gradient_x, local_data_gradient_y, local_data_energy_contribution;

		compute_local_data_term_gradient(warped_live_field, canonical_field, i_col, i_row,
		                                            live_gradient_x_field, live_gradient_y_field,
		                                            local_data_gradient_x, local_data_gradient_y,
		                                            local_data_energy_contribution);


		gradient_field_x(i_element) = local_data_gradient_x;
		gradient_field_y(i_element) = local_data_gradient_y;
	}
}

}// namespace energy_gradient;