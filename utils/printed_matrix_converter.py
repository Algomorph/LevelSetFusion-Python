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


class ConversionMode:
    CppToPython = 0
    PythonToCpp = 1


class MatrixInformation:
    def __init__(self, name, dimensions, element_type, from_line=0, until_line=0):
        self.name = name
        self.dimensions = dimensions
        self.element_type = element_type
        self.from_line = from_line
        self.until_line = until_line
        self.numpy_matrix = None


cpp_type_mappings = {
    "f": float,
    "v2f": float,
    "m2f": float,
    "d": float,
    "i": int
}

cpp_extra_dimension_mappings = {
    "v2f": [2],
    "m2f": [2,2],
}


def main():
    input_file_path = "../input/matrix_converter_input.txt"
    output_directory = "../output/"
    output_filename = "matrix_converter_output"
    conversion_mode = ConversionMode.CppToPython
    if conversion_mode == ConversionMode.CppToPython:
        output_extension = "py"
    elif conversion_mode == ConversionMode.PythonToCpp:
        output_extension = "cpp"
    else:
        raise ValueError("Unsupported ConversionMode: " + str(conversion_mode))
    output_file_path = os.path.join(output_directory, output_filename + "." + output_extension)

    if not os.path.exists(input_file_path) or not os.path.isfile(input_file_path):
        print("Critical error: seems like '{:s}' does not exist or is not a file.".format(input_file_path))

    with open(input_file_path, "r") as input_file:
        lines = input_file.readlines()
        matrix_infos = []
        if conversion_mode == ConversionMode.CppToPython:
            matrix_info_pattern = re.compile(
                r'(?<=MatrixX)(\w+)\s+(\w+)\s*\((\d+),\s+(\d+)\)')
            matrix_header_replacement_pattern = re.compile(
                r'^.*(?:(?:math::Matrix)|(?:eig::Matrix)|(?:Eigen::Matrix)|(?:MatrixX))\w+\s+\w+\s*\(\d+,\s+\d+\);')
            matrix_value_pattern = re.compile(
                r'(?:\s*(?:math::(?:Vector2f|Matrix2f)\()|,\s*)?(-?\d+\.\d*(?:e[+|-]\d\d)?)f?')
            type_mappings = cpp_type_mappings
        elif conversion_mode == ConversionMode.PythonToCpp:
            raise NotImplementedError("PythonToCpp mode not yet implemented!")
        else:
            raise ValueError("Unsupported ConversionMode: " + conversion_mode)

        i_line = 0
        started_reading_matrix = False
        new_lines = []
        for line in lines:
            regex_search_result = re.findall(matrix_info_pattern, line)
            if regex_search_result:
                if len(regex_search_result) > 1:
                    raise ValueError("Multiple matrix declarations on the same line are not supported!")
                regex_search_result = regex_search_result[0]
                if started_reading_matrix:
                    matrix_infos[-1].until_line = i_line
                started_reading_matrix = True
                type_signifier = regex_search_result[0]
                element_type = type_mappings[type_signifier]
                dimensions = (int(regex_search_result[2]), int(regex_search_result[3]))
                # advanced C++ matrix types are multidimensional tensors -- change dimensions accordingly
                if conversion_mode == ConversionMode.CppToPython and type_signifier in cpp_extra_dimension_mappings:
                    dimensions = tuple(list(dimensions) + cpp_extra_dimension_mappings[type_signifier])

                matrix_infos.append(MatrixInformation(name=regex_search_result[1],
                                                      dimensions=dimensions,
                                                      element_type=element_type,
                                                      from_line=i_line))
                # clear out header to avoid value matching to dimensions / header stuff
                line = re.sub(matrix_header_replacement_pattern, "", line)
            new_lines.append(line)
            i_line += 1
        if matrix_infos:
            matrix_infos[-1].until_line = i_line
        lines = new_lines
        for matrix_info in matrix_infos:
            matrix_text = "\n".join(lines[matrix_info.from_line:matrix_info.until_line])
            elements = [matrix_info.element_type(match_result)
                        for match_result in re.findall(matrix_value_pattern, matrix_text)]
            matrix_info.numpy_matrix = np.array(elements).reshape(matrix_info.dimensions)

        output_file = open(output_file_path, "w+")
        if conversion_mode == ConversionMode.CppToPython:
            output_file.write("import numpy as np" + os.linesep)
            for matrix_info in matrix_infos:
                output_file.write(matrix_info.name + " = np." + repr(matrix_info.numpy_matrix) + os.linesep)
        elif conversion_mode == ConversionMode.PythonToCpp:
            raise NotImplementedError("PythonToCpp mode not yet implemented!")
        else:
            raise ValueError("Unsupported ConversionMode: " + str(conversion_mode))

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
