//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>

//local
#include <eigen_numpy.hpp>
#include "nonrigid_optimization/data_term.hpp"
#include "nonrigid_optimization/interpolation.hpp"

namespace bp = boost::python;

Eigen::MatrixXd matrix_product_double(Eigen::MatrixXd a, Eigen::MatrixXd b) {
	return a * b;
}

Eigen::MatrixXf matrix_product_float(Eigen::MatrixXf a, Eigen::MatrixXf b) {
	return a * b;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(interpolate_overloads, interpolation::py_interpolate, 4, 8)

BOOST_PYTHON_MODULE(level_set_fusion_optimization)
{
    SetupEigenConverters();
	bp::def("matrix_product_float64", matrix_product_double);
	bp::def("matrix_product_float32", matrix_product_float);
	bp::def("data_term_at_location", data_term::py_data_term_at_location);
	bp::def("interpolate", interpolation::py_interpolate, interpolate_overloads());
}
