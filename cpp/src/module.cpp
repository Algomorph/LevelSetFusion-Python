//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>

//local
#include <eigen_numpy.hpp>
#include "nonrigid_optimization/data_term.hpp"
#include "nonrigid_optimization/interpolation.hpp"
#include "nonrigid_optimization/sobolev_optimizer2d.hpp"

namespace bp = boost::python;

Eigen::MatrixXd matrix_product_double(Eigen::MatrixXd a, Eigen::MatrixXd b) {
	return a * b;
}

Eigen::MatrixXf matrix_product_float(Eigen::MatrixXf a, Eigen::MatrixXf b) {
	return a * b;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(interpolate_overloads, nonrigid_optimization::py_interpolate, 4, 8)

BOOST_PYTHON_MODULE (level_set_fusion_optimization) {

	namespace nro = nonrigid_optimization;

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

	// classes
	bp::class_<nro::SobolevOptimizer2d>("SobolevOptimizer2d", bp::init<>())
	        .def("optimize", &nro::SobolevOptimizer2d::optimize);


}
