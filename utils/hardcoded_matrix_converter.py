#  ================================================================
#  Created by Gregory Kramida on 11/6/18.
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
# !/usr/bin/python
import sys
import os.path
import re
from enum import Enum

import numpy as np

PROGRAM_EXIT_SUCCESS = 0
PROGRAM_EXIT_FAIL = 1


class ConversionMode(Enum):
    CppToPython = 0
    PythonToCpp = 1


class CppTypes(Enum):
    MatrixXf = "eig::MatrixXf"
    MatrixXd = "eig::MatrixXd"
    MatrixXi = "eig::MatrixXi"
    MatrixXuc = "eig::Matrix<unsigned char, eig::Dynamic, eig::Dynamic>"
    MatrixXus = "eig::Matrix<unsigned short, eig::Dynamic, eig::Dynamic>"
    Tensor3f = "math::Tensor3f"
    MatrixXv2f = "math::MatrixXv2f"
    MatrixXv3f = "math::MatrixXv3f"
    MatrixXm2f = "math::MatrixXm2f"
    MatrixXm3f = "math::MatrixXm3f"
    Tensor3v2f = "math::Tensor3v2f"
    Tensor3v3f = "math::Tensor3v3f"
    Tensor3m2f = "math::Tensor3m2f"
    Tensor3m3f = "math::Tensor3m3f"


class MatrixInformation:
    def __init__(self, name, dimensions, element_type, from_line=0, until_line=0):
        self.name = name
        self.dimensions = dimensions
        self.element_type = element_type
        self.from_line = from_line
        self.until_line = until_line
        self.matrix_type = None
        self.numpy_matrix = None


cpp_to_numpy_type_mappings = {
    "f": np.float32,
    "v2f": np.float32,
    "m2f": np.float32,
    "d": np.float64,
    "i": np.int32
}

cpp_extra_dimension_mappings = {
    "v2f": [2],
    "m2f": [2, 2],
    "v3f": [3]
}

numpy_to_cpp_type_mappings = {
    (1, "no", "float32"): (CppTypes.MatrixXf.value, np.float32),
    (1, "no", "float64"): (CppTypes.MatrixXd.value, np.float64),
    (1, "no", "int32"): (CppTypes.MatrixXi.value, np.int32),
    (1, "no", "uint8"): (CppTypes.MatrixXuc.value, np.uint8),
    (1, "no", "uint16"): (CppTypes.MatrixXus.value, np.uint16),
    (2, "no", "float32"): (CppTypes.MatrixXf.value, np.float32),
    (2, "no", "float64"): (CppTypes.MatrixXd.value, np.float64),
    (2, "no", "int32"): (CppTypes.MatrixXi.value, np.int32),
    (2, "no", "uint8"): (CppTypes.MatrixXi.value, np.uint8),
    (2, "no", "uint16"): (CppTypes.MatrixXuc.value, np.uint16),
    (3, "no", "float32"): (CppTypes.Tensor3f.value, np.float32),
    (3, "v2", "float32"): (CppTypes.MatrixXv2f.value, np.float32),
    (3, "v3", "float32"): (CppTypes.MatrixXv3f.value, np.float32),
    (4, "m2", "float32"): (CppTypes.MatrixXm2f.value, np.float32),
    (4, "m3", "float32"): (CppTypes.MatrixXm3f.value, np.float32),
    (4, "v2", "float32"): (CppTypes.Tensor3v2f.value, np.float32),
    (4, "v3", "float32"): (CppTypes.Tensor3v3f.value, np.float32),
    (5, "m2", "float32"): (CppTypes.Tensor3m2f.value, np.float32),
    (3, "m3", "float32"): (CppTypes.Tensor3m3f.value, np.float32),
}

nested_cpp_types = {
    "v2": [CppTypes.MatrixXv2f.value, CppTypes.Tensor3v2f.value],
    "m2": [CppTypes.MatrixXm2f.value, CppTypes.Tensor3m2f.value],
    "v3": [CppTypes.MatrixXv3f.value, CppTypes.Tensor3v3f.value],
    "m3": [CppTypes.MatrixXm3f.value, CppTypes.Tensor3m3f.value],
}

tensor_cpp_types = {CppTypes.Tensor3f.value, CppTypes.Tensor3m2f.value, CppTypes.Tensor3m3f.value,
                    CppTypes.Tensor3v2f.value, CppTypes.Tensor3v3f.value}


def parse_cpp_header(header):
    type_signifier = header[0]
    element_type = cpp_to_numpy_type_mappings[type_signifier]
    dimensions = (int(header[2]), int(header[3]))
    # advanced C++ matrix types are multidimensional tensors -- change dimensions accordingly
    if type_signifier in cpp_extra_dimension_mappings:
        dimensions = tuple(list(dimensions) + cpp_extra_dimension_mappings[type_signifier])
    return MatrixInformation(name=header[1], dimensions=dimensions, element_type=element_type)


def parse_numpy_header(header):
    # dimensions and element type have to pe parsed later using value regex
    return MatrixInformation(name=header, dimensions=None, element_type=None)


def parse_numpy_dimensions_and_type(value_text, value_count):
    single_bracket_count = len(re.findall(re.compile(r'(?<!\[)\['), value_text))
    double_bracket_count = len(re.findall(re.compile(r'(?<!\[)\[\['), value_text))
    triple_bracket_count = len(re.findall(re.compile(r'(?<!\[)\[\[\['), value_text))
    quadruple_bracket_count = len(re.findall(re.compile(r'(?<!\[)\[\[\[\['), value_text))
    if double_bracket_count == 0:
        # 1D vector
        dimensions = [value_count]
    elif double_bracket_count == 1:
        # 2D matrix
        dimensions = [single_bracket_count, value_count // single_bracket_count]
    elif double_bracket_count > 1:
        # 3D tensor
        if triple_bracket_count == 1:
            row_count = double_bracket_count
            column_count = single_bracket_count // double_bracket_count
            layer_count = value_count // (row_count * column_count)
            dimensions = [row_count, column_count, layer_count]
        else:
            if quadruple_bracket_count > 1:
                # TODO: this should be easy to extend to nested 3D tensors (5-dims) right here. The rest of the code
                #   already supports it.
                raise ValueError("Tensors in more than 4 dimensions are not yet supported.")
            row_count = triple_bracket_count
            column_count = double_bracket_count // triple_bracket_count
            dimension_3 = single_bracket_count // (row_count * column_count)
            dimension_4 = value_count // (row_count * column_count * dimension_3)
            if dimension_4 not in [2, 3]:
                raise ValueError(
                    "Tensors where the fourth dimension is not in {2,3} are "
                    "not currently supported in Python-2-CPP conversion")
            dimensions = [row_count, column_count, dimension_3, dimension_4]
    element_and_matrix_type = ("MatrixXd", np.float64)
    search_result = re.findall(re.compile(r'float32|int32|float64|int34|uint16|uint8'), value_text)
    if search_result:
        nested_subtype = 'no'
        if len(dimensions) in [3, 4, 5]:
            if dimensions[-1] == 2 and dimensions[-2] == 2:
                nested_subtype = 'm2'
            elif dimensions[-1] == 3 and dimensions[-2] == 3:
                nested_subtype = 'm3'
            elif dimensions[-1] == 2:
                nested_subtype = 'v2'
            elif dimensions[-1] == 3:
                nested_subtype = 'v3'
        element_and_matrix_type = numpy_to_cpp_type_mappings[len(dimensions),
                                                             nested_subtype,
                                                             search_result[0]]
    else:
        if len(dimensions) > 2:
            raise ValueError("Tensors of higher dimensions (3,4) need to be explicitly typed. "
                             "Currently, only the float32 type is supported for tensors of higher dimensions than 2.")
    return tuple(dimensions), element_and_matrix_type[1], element_and_matrix_type[0]


# TODO: change architecture so that there are separate MatrixParser subclasses implementing a common interface
# TODO: (to avoid all the mode checks)

def main():
    input_file_path = "../input/matrix_converter_input.txt"
    output_directory = "../output/"
    output_filename = "matrix_converter_output"
    conversion_mode = ConversionMode.CppToPython

    if not os.path.exists(input_file_path) or not os.path.isfile(input_file_path):
        print("Critical error: seems like '{:s}' does not exist or is not a file.".format(input_file_path))

    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()
        for line in lines:
            if "MatrixXf" in line:
                conversion_mode = ConversionMode.CppToPython
                break
            elif "np.array" in line or "numpy.array" in line:
                conversion_mode = ConversionMode.PythonToCpp
                break
        if conversion_mode == ConversionMode.CppToPython:
            output_extension = "py"
        elif conversion_mode == ConversionMode.PythonToCpp:
            output_extension = "cpp"
        else:
            raise ValueError("Unsupported ConversionMode: " + str(conversion_mode))
        output_file_path = os.path.join(output_directory, output_filename + "." + output_extension)

        matrix_infos = []
        if conversion_mode == ConversionMode.CppToPython:
            matrix_header_pattern = re.compile(
                r'(?<=MatrixX)(\w+)\s+(\w+)\s*\((\d+),\s+(\d+)\)')
            matrix_header_replacement_pattern = re.compile(
                r'^.*(?:(?:math::Matrix)|(?:eig::Matrix)|(?:Eigen::Matrix)|(?:MatrixX))\w+\s+\w+\s*\(\d+,\s+\d+\);')
            matrix_value_pattern = re.compile(
                r'(?:\s*(?:math::(?:Vector2f|Matrix2f)\()|,\s*)?(-?\d+\.?\d*(?:e[+|-]\d\d)?)f?\s*(?:,|\))')
        elif conversion_mode == ConversionMode.PythonToCpp:
            matrix_header_replacement_pattern = \
                matrix_header_pattern = re.compile(r'(\w+)\s*=\s*(?:np|numpy)\.array\(')
            matrix_value_pattern = re.compile(r'(?:\s*((?<!float)(?<!int)-?\d+\.?\d*(?:e[+|-]\d\d)?(?!\)))\s*(?:\]|,))')
        else:
            raise ValueError("Unsupported ConversionMode: " + str(conversion_mode))

        i_line = 0
        started_reading_matrix = False
        new_lines = []
        for line in lines:
            regex_search_result = re.findall(matrix_header_pattern, line)
            if regex_search_result:
                if len(regex_search_result) > 1:
                    raise ValueError("Multiple matrix declarations on the same line are not supported!")
                if started_reading_matrix:
                    matrix_infos[-1].until_line = i_line
                started_reading_matrix = True
                if conversion_mode == ConversionMode.CppToPython:
                    matrix_info = parse_cpp_header(regex_search_result[0])
                elif conversion_mode == ConversionMode.PythonToCpp:
                    matrix_info = parse_numpy_header(regex_search_result[0])
                else:
                    raise ValueError("Unsupported ConversionMode: " + str(conversion_mode))

                matrix_info.from_line = i_line
                matrix_infos.append(matrix_info)

                # clear out header to avoid value matching to header stuff in some situations
                line = re.sub(matrix_header_replacement_pattern, "", line)
            new_lines.append(line)
            i_line += 1
        if matrix_infos:
            matrix_infos[-1].until_line = i_line
        lines = new_lines

        if len(matrix_infos) == 1:
            print("Found 1 matrix")
        else:
            print("Found {:d} matrices".format(len(matrix_infos)))
        for matrix_info in matrix_infos:
            matrix_text = "".join(lines[matrix_info.from_line:matrix_info.until_line])
            if conversion_mode == ConversionMode.CppToPython:
                elements = [matrix_info.element_type(match_result)
                            for match_result in re.findall(matrix_value_pattern, matrix_text)]
                matrix_info.numpy_matrix = np.array(elements).reshape(matrix_info.dimensions)
            elif conversion_mode == ConversionMode.PythonToCpp:
                element_strings = re.findall(matrix_value_pattern, matrix_text)
                matrix_info.dimensions, matrix_info.element_type, matrix_info.matrix_type = \
                    parse_numpy_dimensions_and_type(matrix_text, len(element_strings))
                matrix_info.numpy_matrix = \
                    np.array([matrix_info.element_type(element_strings)]).reshape(matrix_info.dimensions)
            else:
                raise ValueError("Unsupported ConversionMode: " + str(conversion_mode))

        output_file = open(output_file_path, "w+")
        if conversion_mode == ConversionMode.CppToPython:
            output_file.write("import numpy as np" + os.linesep + os.linesep)
            for matrix_info in matrix_infos:
                output_file.write(matrix_info.name + " = np." +
                                  repr(matrix_info.numpy_matrix).replace("float32", "np.float32")
                                  .replace("int32", "np.int32") + os.linesep)

        elif conversion_mode == ConversionMode.PythonToCpp:
            static_global_variables = True
            output_file.write("#include <Eigen/Eigen>" + os.linesep)
            output_file.write("#include \"../math/tensors.hpp\"" + os.linesep + os.linesep)
            output_file.write("namespace eig=Eigen;")
            for matrix_info in matrix_infos:
                output_file.write(os.linesep + os.linesep)
                element_suffix = ""
                if "f" in matrix_info.matrix_type:
                    element_suffix = "f"

                if matrix_info.matrix_type in tensor_cpp_types:
                    # Eigen Tensors default to column-major order, so z,y,x becomes x,y,z
                    dimension_string = str(matrix_info.dimensions[2]) + "," + str(
                        matrix_info.dimensions[1]) + "," + str(matrix_info.dimensions[0])
                else:
                    dimension_string = str(matrix_info.dimensions[0]) + "," + str(matrix_info.dimensions[1])

                line_prefix_whitespace = ""
                if static_global_variables:
                    line_prefix_whitespace = "		"
                    output_file.write(
                        "static " + matrix_info.matrix_type + " " + matrix_info.name + " = []{" + os.linesep)

                output_file.write(line_prefix_whitespace + matrix_info.matrix_type + " " +
                                  matrix_info.name + "(" + dimension_string + ");" + os.linesep)
                nested = True

                esx = element_suffix
                if matrix_info.matrix_type in nested_cpp_types['v2']:
                    elements = matrix_info.numpy_matrix.flatten().reshape(-1, 2)

                    def write_element(elem):
                        return str(
                            "math::Vector2f(" + str(elem[0]) + esx + "," + str(elem[1]) + esx + ")")

                elif matrix_info.matrix_type in nested_cpp_types['v3']:
                    elements = matrix_info.numpy_matrix.flatten().reshape(-1, 3)

                    def write_element(elem):
                        return str(
                            "math::Vector3f("
                            + str(elem[0]) + esx + "," + str(elem[1]) + esx + "," + str(elem[2]) + esx + ")")

                elif matrix_info.matrix_type in nested_cpp_types['m2']:
                    elements = matrix_info.numpy_matrix.flatten().reshape(-1, 2, 2)

                    def write_element(elem):
                        return str(
                            "math::Matrix2f(" +
                            str(elem[0, 0]) + esx + "," + str(elem[0, 1]) + esx + "," +
                            str(elem[1, 0]) + esx + "," + str(elem[1, 1]) + esx + ")")
                elif matrix_info.matrix_type in nested_cpp_types['m3']:
                    elements = matrix_info.numpy_matrix.flatten().reshape(-1, 3, 3)

                    def write_element(elem):
                        return str(
                            "math::Matrix3f(" +
                            str(elem[0, 0]) + esx + "," + str(elem[0, 1]) + esx + "," + str(elem[0, 2]) + esx + "," +
                            str(elem[1, 0]) + esx + "," + str(elem[1, 1]) + esx + "," + str(elem[1, 2]) + esx + "," +
                            str(elem[2, 0]) + esx + "," + str(elem[2, 1]) + esx + "," + str(elem[2, 2]) + esx + ")")
                else:
                    elements = matrix_info.numpy_matrix.flatten()
                    nested = False

                    def write_element(elem):
                        return str(elem) + element_suffix

                if matrix_info.matrix_type in tensor_cpp_types:
                    output_file.write(line_prefix_whitespace + matrix_info.name + ".setValues(  // @formatter:off" +
                                      os.linesep + line_prefix_whitespace + "{")
                    for x in range(matrix_info.dimensions[2]):
                        if x > 0:
                            output_file.write(line_prefix_whitespace + " ")
                        output_file.write("{")
                        for y in range(matrix_info.dimensions[1]):
                            if y > 0:
                                output_file.write(line_prefix_whitespace + "  ")
                            output_file.write("{")
                            for z in range(matrix_info.dimensions[0]):
                                element = matrix_info.numpy_matrix[z, y, x]
                                if z < matrix_info.dimensions[0] - 1:
                                    output_file.write(write_element(element) + ", ")
                                else:
                                    output_file.write(write_element(element))
                            if y < matrix_info.dimensions[1] - 1:
                                output_file.write("}," + os.linesep)
                            else:
                                output_file.write("}")
                        if x < matrix_info.dimensions[2] - 1:
                            output_file.write("}," + os.linesep)
                        else:
                            output_file.write("}")
                    output_file.write("}); // @formatter:on")
                else:
                    output_file.write(line_prefix_whitespace + matrix_info.name + " << ")
                    i_element = 0
                    for element in elements[:-1]:
                        if nested and not i_element == 0:
                            output_file.write(os.linesep + line_prefix_whitespace)
                        if i_element % matrix_info.dimensions[1] == 0:
                            output_file.write(os.linesep + line_prefix_whitespace)
                        output_file.write(write_element(element) + ", ")
                        i_element += 1
                    if nested:
                        output_file.write(os.linesep + line_prefix_whitespace)
                    output_file.write(write_element(elements[-1]) + ";")
                if static_global_variables:
                    output_file.write(os.linesep + line_prefix_whitespace + "return " + matrix_info.name + ";" +
                                      os.linesep + "}();")

        else:
            raise ValueError("Unsupported ConversionMode: " + str(conversion_mode))

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
