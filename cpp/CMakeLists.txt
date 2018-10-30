#  ================================================================
#  Created by Gregory Kramida on 10/10/18.
#  Copyright (c) 2018 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================

cmake_minimum_required(VERSION 3.9)
project(LevelSetFusionOptimization LANGUAGES CXX)
# C++11 standard simplifies things in a lot of obvious ways,
# it also allows us to use Eigen in multithreaded code w/o special initialization
set(CMAKE_CXX_STANDARD 11)

# Name of the target and python module
set(MODULE_NAME level_set_fusion_optimization)

# Put our local cmake find scripts at the beginning of the CMake
# module search path.
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#=================================================================
# PYTHON option

set(PYTHON_OPTIONS "2.X" "3.X")
set(PYTHON_DESIRED_VERSION "3.X" CACHE STRING "Choose which python version to use, options are: ${PYTHON_OPTIONS}.")
set_property(CACHE PYTHON_DESIRED_VERSION PROPERTY STRINGS ${PYTHON_OPTIONS})

#==================== FIND PACKAGES ==============================

# *** Python
include("DetectPython")

if (NOT BOOST_PYTHON_NUMPY_EIGEN_PYTHON_VERSION)
    set(BOOST_PYTHON_NUMPY_EIGEN_PYTHON_VERSION 3.4 CACHE STRING "Choose the target Python version (e.g., 3.4, 2.7)" FORCE)
endif ()

include_directories(include)

# *** Boost (Boost python version based on python option / availability)
if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    # Provisions for typical Boost compiled on Windows
    # Most commonly, Boost libraries are compiled statically on windows (change as necesssary)
    set(Boost_USE_STATIC_LIBS TRUE)
    set(Boost_USE_STATIC_RUNTIME OFF)
    set(Boost_USE_MULTITHREADED ON)
    set(Boost_USE_DEBUG_RUNTIME ON)
    set(Boost_USE_DEBUG_PYTHON OFF)
endif()
set(BOOST_COMPONENTS unit_test_framework)
if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    set(Python_ADDITIONAL_VERSIONS ${PYTHON2_VERSION_MAJOR}.${PYTHON2_VERSION_MINOR})
    list(APPEND BOOST_COMPONENTS python${PYTHON2_VERSION_MAJOR}${PYTHON2_VERSION_MINOR})
    set(PYTHON_EXECUTABLE ${Python2_EXECUTABLE})
else ()
    set(Python_ADDITIONAL_VERSIONS ${PYTHON3_VERSION_MAJOR}.${PYTHON3_VERSION_MINOR})
    list(APPEND BOOST_COMPONENTS python${PYTHON3_VERSION_MAJOR}${PYTHON3_VERSION_MINOR})
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif ()


find_package(Boost COMPONENTS ${BOOST_COMPONENTS} REQUIRED)
# *** Everything else
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

#======== PICK PYTHON STUFF========================================
if (${PYTHON_DESIRED_VERSION} STREQUAL "2.X")
    set(PYTHON_INCLUDE_DIRS ${PYTHON2_INCLUDE_DIR} ${PYTHON2_INCLUDE_DIR2} ${PYTHON2_NUMPY_INCLUDE_DIRS})
    set(PYTHON_NUMPY_INCLUDE_DIRS ${PYTHON2_NUMPY_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${PYTHON2_LIBRARY})
    set(PYTHON_EXECUTABLE ${PYTHON2_EXECUTABLE})
    set(PYTHON_PACKAGES_PATH ${PYTHON2_PACKAGES_PATH})
    set(ARCHIVE_OUTPUT_NAME ${MODULE_NAME}_py2)
else ()
    set(PYTHON_INCLUDE_DIRS ${PYTHON3_INCLUDE_DIR} ${PYTHON3_INCLUDE_DIR2} ${PYTHON3_NUMPY_INCLUDE_DIRS})
    set(PYTHON_NUMPY_INCLUDE_DIRS ${PYTHON3_NUMPY_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${PYTHON3_LIBRARY})
    set(PYTHON_EXECUTABLE ${PYTHON3_EXECUTABLE})
    set(PYTHON_PACKAGES_PATH ${PYTHON3_PACKAGES_PATH})
    set(ARCHIVE_OUTPUT_NAME ${MODULE_NAME}_py3)
endif ()
find_package_handle_standard_args(Python DEFAULT_MSG PYTHON_INCLUDE_DIRS PYTHON_NUMPY_INCLUDE_DIRS PYTHON_LIBRARIES PYTHON_EXECUTABLE PYTHON_PACKAGES_PATH)
if(NOT Python_FOUND)
    message(SEND_ERROR "Not all requred components of Numpy/Python found.")
endif()

# Find true site-packages site on this machine, since find_package(PythonLibs) doesn't seem to get the correct one
# (Adapted from OpenCV3)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
        "from distutils.sysconfig import get_python_lib;\
  print(get_python_lib(plat_specific=True, prefix=''))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )

set(MODULE_INSTALL_PREFIX "${MODULE_INSTALL_PREFIX}" CACHE STRING
        "Output directory for Python modules.")


#-get proper extension for python binary shared object on this platform
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SO'))"
        RESULT_VARIABLE PYTHON_${MODULE_NAME}_PY_PROCESS
        OUTPUT_VARIABLE ${MODULE_NAME}_PY_SUFFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE)


#=================== INCLUDE PATHS =====================================================================================

# Global settings for include paths.
include_directories(
        ${PYTHON_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${NUMPY_INCLUDE_DIRS}
)

#=================== TARGET DEFININTIONS ===============================================================================

# main library
add_library(${MODULE_NAME} SHARED
            include/eigen_numpy.hpp
            src/eigen_numpy.cpp
            src/module.cpp
            src/nonrigid_optimization/data_term.hpp
            src/nonrigid_optimization/data_term.cpp
            src/nonrigid_optimization/interpolation.hpp
            src/nonrigid_optimization/interpolation.cpp
            src/nonrigid_optimization/boolean_operations.hpp
            src/nonrigid_optimization/full_gradient.cpp
            src/nonrigid_optimization/full_gradient.hpp
            src/nonrigid_optimization/smoothing_term.cpp
            src/nonrigid_optimization/smoothing_term.hpp
            src/math/tensors.hpp
            src/math/tensors.cpp
            src/math/math_utils.hpp
            src/math/vector_base.hpp
            src/math/matrix_base.hpp
            src/math/vector2.hpp
            src/math/platform_independence.hpp
            src/math/matrix2.hpp src/math/vector3.hpp
            src/math/typedefs.hpp
            src/math/conversion.hpp src/math/gradients.cpp src/math/gradients.hpp src/traversal/field_traversal_cpu.hpp)
target_link_libraries(${MODULE_NAME}
        PUBLIC
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES})

target_link_libraries(${MODULE_NAME}
        PRIVATE
        OpenMP::OpenMP_CXX)

if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
    # Provisions for typical Boost compiled on Windows
    # Unless some extra compile options are used on Windows, the libraries won't have prefixes (change as necesssary)
    target_compile_definitions(${MODULE_NAME} PUBLIC -DBOOST_ALL_NO_LIB -DBOOST_SYSTEM_NO_DEPRECATED)
else()
    target_compile_definitions(${MODULE_NAME} PUBLIC -DBOOST_TEST_DYN_LINK)
endif()

#message(STATUS "${MODULE_NAME}: [archive output name:]  ${ARCHIVE_OUTPUT_NAME} [archive suffix:] ${${MODULE_NAME}_PY_SUFFIX}")
#message(STATUS "[include dirs:] ${PYTHON_INCLUDE_DIRS} [packages path:] ${PYTHON3_PACKAGES_PATH} [libraries:] ${PYTHON_LIBRARIES}")

#======== INSTALLATION MAKE TARGET =====================================================================================

set_target_properties(${MODULE_NAME} PROPERTIES
        ARCHIVE_OUTPUT_NAME ${ARCHIVE_OUTPUT_NAME}  # prevent name conflict for python2/3 outputs
        PREFIX ""
        OUTPUT_NAME ${MODULE_NAME}
        SUFFIX ${${MODULE_NAME}_PY_SUFFIX})


if ((CMAKE_CXX_COMPILER_ID MATCHES MSVC) AND (NOT PYTHON_DEBUG_LIBRARIES))
    set(PYTHON_INSTALL_CONFIGURATIONS CONFIGURATIONS Release)
else ()
    set(PYTHON_INSTALL_CONFIGURATIONS "")
endif ()

if (WIN32)
    set(PYTHON_INSTALL_ARCHIVE "")
else ()
    set(PYTHON_INSTALL_ARCHIVE ARCHIVE DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python)
endif ()

install(TARGETS ${MODULE_NAME}
        ${PYTHON_INSTALL_CONFIGURATIONS}
        RUNTIME DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        LIBRARY DESTINATION ${PYTHON_PACKAGES_PATH} COMPONENT python
        ${PYTHON_INSTALL_ARCHIVE}
        )
# ================== UNIT TESTING ======================================================================================

enable_testing()

add_executable(test_nonrigid_optimization tests/test_nonrigid_optimization.cpp)
add_executable(test_math tests/test_math.cpp tests/test_data_math.hpp tests/test_data_nonrigid_optimization.hpp)

target_link_libraries(test_nonrigid_optimization ${MODULE_NAME})
target_link_libraries(test_math ${MODULE_NAME})

add_test(NAME nonrigid_optimization COMMAND test_nonrigid_optimization)
add_test(NAME math COMMAND test_math)
