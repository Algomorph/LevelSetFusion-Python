//stdlib
#include <iostream>

//libraries
#include <Eigen/Eigen>
#include <boost/python.hpp>

//local
#include <eigen_numpy.hpp>

namespace bp = boost::python;
static const int X = Eigen::Dynamic;

Eigen::Matrix<double,X,X> mul(Eigen::Matrix<double,X,X> a, Eigen::Matrix<double,X,X> b) {
	return a * b;
}

BOOST_PYTHON_MODULE(level_set_fusion_optimization)
{
    SetupEigenConverters();
	bp::def("mul", mul);


}
