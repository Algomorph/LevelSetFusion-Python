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
#include "../traversal/field_traversal_cpu.hpp"

namespace nonrigid_optimization {


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
struct DataTermGradientAndEnergyFunctor {
	float data_term_energy = 0.0f;
	float scaling_factor = 10.0f;

	DataTermGradientAndEnergyFunctor(float scaling_factor = 10.0f) {
		this->scaling_factor = scaling_factor;
	}

	inline math::Vector2f operator()(const float& live_tsdf_value,
	                                 const float& canonical_tsdf_value,
	                                 const math::Vector2f& local_live_gradient) {
		if (TSkipTruncated) {
			if (is_outside_narrow_band(live_tsdf_value, canonical_tsdf_value)) {
				return math::Vector2f(0.0f);
			}
		}
		float diff = live_tsdf_value - canonical_tsdf_value;
		data_term_energy += .5 * diff * diff;
		return scaling_factor * diff * local_live_gradient;
	}
};

/**
 * \brief Computes the gradient of the data energy term for KillingFusion/SobolevFusion-based optimization on a 2D grid
 * \details Goes over every location, regardless of whether the TSDF values are within the narrow band (are not truncated)
 * See Section 4.1 in KillingFusion[1] / 1.1 in KillingFusion Supplementary Material / 4.1 in SobolevFusion[2]
 * [1] M. Slavcheva, M. Baust, D. Cremers, and S. Ilic, “KillingFusion: Non-rigid 3D Reconstruction without Correspondences,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, no. 4, pp. 1386–1395.
 * [2] M. Slavcheva, M. Baust, and S. Ilic, “SobolevFusion : 3D Reconstruction of Scenes Undergoing Free Non-rigid Motion,” in Computer Vision and
 *
 * \param[out] data_term_gradient output gradient
 * \param[out] data_term_energy output energy
 * \param warped_live_field 2D TSBF representing the warped data from a new frame
 * \param canonical_field 2D TSDF representing the canonical (fused) data
 * \param warped_live_field_gradient a field with vector gradients of the warped new-frame data
 * \param[in] scaling_factor factor to scale the gradient. Usually, narrow-band half-width divided by the voxel size.
 */
void compute_data_term_gradient(
		math::MatrixXv2f& data_term_gradient, float& data_term_energy,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		const math::MatrixXv2f& warped_live_field_gradient, float scaling_factor) {
	DataTermGradientAndEnergyFunctor<false> functor(scaling_factor);
	data_term_gradient = math::MatrixXv2f(warped_live_field.rows(), warped_live_field.cols());
	traversal::traverse_triple_2d_field_output_field_singlethreaded(data_term_gradient, warped_live_field,
	                                                                canonical_field, warped_live_field_gradient,
	                                                                functor);
	data_term_energy = functor.data_term_energy;
}

/**
 * \brief Does the same thing as compute_data_term_gradient(), except returns a zero vector as gradient for any locations
 * where both the passed warped live field and canonical field values are truncated
 * \param[out] data_term_gradient output gradient
 * \param[out] data_term_energy output energy
 * \param warped_live_field 2D TSBF representing the warped data from a new frame
 * \param canonical_field 2D TSDF representing the canonical (fused) data
 * \param warped_live_field_gradient a field with vector gradients of the warped new-frame data
 * \param[in] scaling_factor factor to scale the gradient. Usually, narrow-band half-width divided by the voxel size.
 */
void compute_data_term_gradient_within_band_union(
		math::MatrixXv2f& data_term_gradient, float& data_term_energy,
		const eig::MatrixXf& warped_live_field, const eig::MatrixXf& canonical_field,
		const math::MatrixXv2f& warped_live_field_gradient, float scaling_factor) {
	DataTermGradientAndEnergyFunctor<true> functor(scaling_factor);
	data_term_gradient = math::MatrixXv2f(warped_live_field.rows(), warped_live_field.cols());
	traversal::traverse_triple_2d_field_output_field_singlethreaded(data_term_gradient, warped_live_field,
	                                                                canonical_field, warped_live_field_gradient,
	                                                                functor);
	data_term_energy = functor.data_term_energy;
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


}//namespace nonrigid_optimization
