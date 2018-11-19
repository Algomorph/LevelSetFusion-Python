//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

//local
#include <eigen_numpy.hpp>
#include "nonrigid_optimization/data_term.hpp"
#include "nonrigid_optimization/interpolation.hpp"
#include "nonrigid_optimization/sobolev_optimizer2d.hpp"
#include "nonrigid_optimization/logging.hpp"

namespace bp = boost::python;
namespace nro = nonrigid_optimization;

typedef nro::SobolevOptimizer2d::SobolevParameters SobolevParameters;
typedef nro::Optimizer2d::SharedParameters SharedParameters;

Eigen::MatrixXd matrix_product_double(Eigen::MatrixXd a, Eigen::MatrixXd b) {
	return a * b;
}

Eigen::MatrixXf matrix_product_float(Eigen::MatrixXf a, Eigen::MatrixXf b) {
	return a * b;
}

struct NullDeleter {
	void operator()(const void*) {
	}
};

boost::shared_ptr<SobolevParameters> get_shared_sobolev_parameters() {
	return boost::shared_ptr<SobolevParameters>(
			&SobolevParameters::get_instance(), NullDeleter());
}

boost::shared_ptr<SharedParameters> get_shared_shared_parameters() {
	return boost::shared_ptr<SharedParameters>(&SharedParameters::get_instance(), NullDeleter());
}

BOOST_PYTHON_FUNCTION_OVERLOADS(interpolate_overloads, nonrigid_optimization::py_interpolate, 4, 8)

BOOST_PYTHON_MODULE (level_set_fusion_optimization)
{
	SetupEigenConverters();

	// functions
	bp::def("matrix_product_float64", matrix_product_double);
	bp::def("matrix_product_float32", matrix_product_float);
	bp::def("data_term_at_location", nro::py_data_term_at_location);
	bp::def("interpolate", nro::py_interpolate,
			interpolate_overloads(
					bp::args("warped_live_field", "canonical_field", "warp_field_u", "warp_field_v", "band_union_only",
							"known_values_only", "substitute_original", "truncation_float_threshold"),
					"interpolate (warp) a 2D scalar field using a 2D vector field, such that new value at (x,y)"
							" is interpolated bilinearly from the original scalar field (warped_live_field) at the location "
							"(x+u, y+v), where"
							" the vector (u,v) is located at (x,y) within the warp field (i.e. u = warp_field_u[y,x]).\n"
							" :param band_union_only when set to True, skips processing locations for which both"
							" warped_live_field and canonical_field contain the values +1 or -1 (truncated sdf values)")
					);
	// =============================================================================================
	// region === MATH TYPES =======================================================================
	// =============================================================================================
	bp::class_<math::Vector2i>("Vector2i", bp::init<int, int>())
			.def(bp::init<int>())
			.def_readwrite("x", &math::Vector2i::x)
			.def_readwrite("y", &math::Vector2i::y)
			.def_readwrite("u", &math::Vector2i::u)
			.def_readwrite("v", &math::Vector2i::v)
			;

	bp::class_<math::Vector2f>("Vector2f", bp::init<float, float>())
			.def(bp::init<float>())
			.def_readwrite("x", &math::Vector2f::x)
			.def_readwrite("y", &math::Vector2f::y)
			.def_readwrite("u", &math::Vector2f::u)
			.def_readwrite("v", &math::Vector2f::v)
			;

	// endregion ===================================================================================
	// =============================================================================================
	// region === PARAMETER SINGLETON TYPES ========================================================
	// =============================================================================================

	bp::class_<SobolevParameters, boost::shared_ptr<SobolevParameters>, boost::noncopyable>(
			"SobolevParameters", bp::no_init)
			.def("get_instance", &get_shared_sobolev_parameters)
			.staticmethod("get_instance")
			//.def_readwrite("sobolev_kernel", &SobolevParameters::sobolev_kernel)
			.def("get_sobolev_kernel", &SobolevParameters::get_sobolev_kernel)
			.def("set_sobolev_kernel", &SobolevParameters::set_sobolev_kernel)
			.def_readwrite("smoothing_term_weight", &SobolevParameters::smoothing_term_weight);

	bp::class_<SharedParameters, boost::shared_ptr<SharedParameters>, boost::noncopyable>(
			"SharedParameters", bp::no_init)
			.def("get_instance", &get_shared_shared_parameters)
			.staticmethod("get_instance")

	.def_readwrite("gradient_descent_rate", &SharedParameters::gradient_descent_rate)

	//termination condition parameters
	.def_readwrite("maximum_warp_length_lower_threashold",
			&SharedParameters::maximum_warp_length_lower_threshold)
			.def_readwrite("maximum_warp_length_upper_threashold",
			&SharedParameters::maximum_warp_length_upper_threshold)
			.def_readwrite("maximum_iteration_count", &SharedParameters::maximum_iteration_count)
			.def_readwrite("minimum_iteration_count", &SharedParameters::minimum_iteration_count)

	//logging
	.def_readwrite("enable_focus_spot_analytics", &SharedParameters::enable_focus_spot_analytics)
			.def_readwrite("enable_convergence_status_logging",
			&SharedParameters::enable_convergence_status_logging)
			.def_readwrite("enable_live_sdf_progression_logging",
			&SharedParameters::enable_live_sdf_progression_logging)
			.def_readwrite("enable_gradient_logging", &SharedParameters::enable_gradient_logging)
			.def_readwrite("enable_gradient_component_logging", &SharedParameters::enable_gradient_component_logging)
			.def_readwrite("enable_warp_statistics_logging", &SharedParameters::enable_warp_statistics_logging)

	//focus spot
	//TODO: math::Vector<X> C++-->numpy converter is needed for this
	.def_readwrite("focus_spot", &SharedParameters::focus_spot);
	// endregion ===================================================================================

	// =============================================================================================
	// region === Logging ==========================================================================
	// =============================================================================================
	bp::class_<nro::ConvergenceStatus>("ConvergenceStatus", bp::init<>())
			.def(bp::init<int, float, math::Vector2i, bool, bool, bool>())
			.def_readwrite("iteration_count", &nro::ConvergenceStatus::iteration_count)
			.def_readwrite("max_warp_length", &nro::ConvergenceStatus::max_warp_length)
			.def_readwrite("max_warp_location", &nro::ConvergenceStatus::max_warp_location)
			.def_readwrite("iteration_limit_reached", &nro::ConvergenceStatus::iteration_limit_reached)
			.def_readwrite("largest_warp_below_minimum_threshold",
			&nro::ConvergenceStatus::largest_warp_below_minimum_threshold)
			.def_readwrite("largest_warp_above_maximum_threshold",
			&nro::ConvergenceStatus::largest_warp_above_maximum_threshold)
			;

	bp::class_<nro::IterationWarpStatistics>("WarpStatistics", bp::init<>())
			.def(bp::init<float,float,float,float,math::Vector2i>())
			.def_readwrite("ratio_of_warps_above_minimum_threshold",
					&nro::IterationWarpStatistics::ratio_of_warps_above_minimum_threshold)
			.def_readwrite("max_warp_length", &nro::IterationWarpStatistics::max_warp_length)
			.def_readwrite("average_warp_length", &nro::IterationWarpStatistics::mean_warp_length)
			.def_readwrite("standard_deviation_of_warp_length",
					&nro::IterationWarpStatistics::standard_deviation_of_warp_length)
			.def_readwrite("max_warp_location", &nro::IterationWarpStatistics::max_warp_location)
			.def("to_array", &nro::IterationWarpStatistics::to_array)
			;

	// endregion ===================================================================================

	// =============================================================================================
	// region === Algorithms =======================================================================
	// =============================================================================================

	bp::class_<nro::SobolevOptimizer2d>("SobolevOptimizer2d", bp::init<>())
			.def("optimize", &nro::SobolevOptimizer2d::optimize)
			.def("get_convergence_status", &nro::SobolevOptimizer2d::get_convergence_status)
			.def("get_warp_statistics_as_matrix", &nro::SobolevOptimizer2d::get_warp_statistics_as_matrix)
			;
	// endregion
}
