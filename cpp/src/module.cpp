//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>

//local
#include <eigen_numpy.hpp>
#include "nonrigid_optimization/DataTermComputer.hpp"

namespace bp = boost::python;

Eigen::MatrixXd mul(Eigen::MatrixXd a, Eigen::MatrixXd b) {
	return a * b;
}

Eigen::MatrixXf mulf(Eigen::MatrixXf a, Eigen::MatrixXf b) {
	return a * b;
}

BOOST_PYTHON_MODULE(level_set_fusion_optimization)
{
    SetupEigenConverters();
	bp::def("mul", mul);
	bp::def("mulf", mulf);
	bp::def("data_term_at_location", data_term::data_term_at_location);

}
