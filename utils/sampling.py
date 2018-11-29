#  ================================================================
#  Created by Gregory Kramida on 9/17/18.
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

# routines that sample from a given scalar or vector field at the specified location in various ways

# stdlib
import math
# libraries
import numpy as np
# local
from utils.point import Point


# TODO: move all the focus coordinate stuff to a separate file
FOCUS_COORDINATES = (33, 76)


def set_focus_coordinates(x, y):
    global FOCUS_COORDINATES
    FOCUS_COORDINATES = (x, y)


def sample_at(field, x=0, y=0, point=None):
    """
    Sample from a 2D scalar field at a given coordinate; return 1 if given coordinate is out-of-bounds
    Works with either a named point argument or x and y coordinates positionally following the filed argument.
    In case all three are specified and point is not None, it will override the x and y arguments.
    :param field: field from which to sample
    :type field: numpy.ndarray
    :param x: x coordinate for sampling location
    :type x: int
    :param y: y coordinate for sampling location
    :type y: int
    :param point: full coordinate for sampling location.
    :type point: Point
    :return: scalar value at given coordinate if (x,y) are within bounds of the scalar field, 1 otherwise
    """
    if point is not None:
        x = point.x
        y = point.y
    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        return 1
    return field[y, x]


def sample_flag_at(field, x=0, y=0, point=None):
    if point is not None:
        x = point.x
        y = point.y
    if x < 0.0 or x >= field.shape[1] or y < 0.0 or y >= field.shape[0]:
        return 0
    return field[y, x]


def sample_at_replacement(field, replacement, x=0, y=0, point=None):
    if point is not None:
        x = point.x
        y = point.y
    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        return replacement
    return field[y, x]


def focus_coordinates_match(x, y):
    return x == FOCUS_COORDINATES[0] and y == FOCUS_COORDINATES[1]


def get_focus_coordinates():
    return FOCUS_COORDINATES


def sample_warp(warp_field, x, y, replacement):
    if x >= warp_field.shape[1] or y >= warp_field.shape[0] or x < 0 or y < 0:
        return replacement
    else:
        return warp_field[y, x]


def sample_warp_replace_if_zero(warp_field, x, y, replacement):
    if x >= warp_field.shape[1] or y >= warp_field.shape[0] or x < 0 or y < 0 or np.linalg.norm(
            warp_field[y, x]) == 0.0:
        return replacement
    else:
        return warp_field[y, x]


class BilinearSamplingMetaInfo():
    def __init__(self, value00=1, value01=1, value10=1, value11=1, ratios=Point(1.0, 1.0),
                 inverse_ratios=Point(0.0, 0.0)):
        self.value00 = value00
        self.value01 = value01
        self.value10 = value10
        self.value11 = value11
        self.ratios = ratios
        self.inverse_ratios = inverse_ratios


# The coordinate is considered out-of-bounds if it is further than 1 cell away from the border of the field.
#     If the coordinate is outside of the field border but within 1 cell, the border values are repeated to substitute
#      for missing values in the interpolation process.
def bilinear_sample_at(field, x=0, y=0, point=None):
    """
    Sample from a 2D scalar field at a given coordinate with bilinear interpolation.
    If coordinate is out-of-bounds, uses "1" for all samples that fall out of bounds during the interpolation.
    Works with either a named point argument or x and y coordinates positionally following the filed argument.
    In case all three are specified and point is not None, it will override the x and y arguments.
    :param field: field from which to sample
    :type field: numpy.ndarray
    :param x: x coordinate for sampling location
    :type x: int
    :param y: y coordinate for sampling location
    :type y: int
    :param point: full coordinate for sampling location.
    :type point: Point
    :return: bilinearly interpolated scalar value at given coordinate if (x,y) are within bounds of the scalar field,
     1 otherwise
    """
    if point is not None:
        x = point.x
        y = point.y

    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        return 1

    point = Point(x, y)

    base_point = Point(math.floor(point.x), math.floor(point.y))
    ratios = point - base_point
    inverse_ratios = Point(1.0, 1.0) - ratios

    value00 = sample_at(field, point=base_point)
    value01 = sample_at(field, point=base_point + Point(0, 1))
    value10 = sample_at(field, point=base_point + Point(1, 0))
    value11 = sample_at(field, point=base_point + Point(1, 1))

    interpolated_value0 = value00 * inverse_ratios.y + value01 * ratios.y
    interpolated_value1 = value10 * inverse_ratios.y + value11 * ratios.y
    interpolated_value = interpolated_value0 * inverse_ratios.x + interpolated_value1 * ratios.x

    return interpolated_value


# The coordinate is considered out-of-bounds if it is further than 1 cell away from the border of the field.
#     If the coordinate is outside of the field border but within 1 cell, the border values are repeated to substitute
#      for missing values in the interpolation process.
def bilinear_sample_at_metainfo(field, x=0, y=0, point=None):
    """
    Sample from a 2D scalar field at a given coordinate with bilinear interpolation.
    If coordinate is out-of-bounds, uses "1" for all samples that fall out of bounds during the interpolation.
    Works with either a named point argument or x and y coordinates positionally following the filed argument.
    In case all three are specified and point is not None, it will override the x and y arguments.
    :param field: field from which to sample
    :type field: numpy.ndarray
    :param x: x coordinate for sampling location
    :type x: int
    :param y: y coordinate for sampling location
    :type y: int
    :param point: full coordinate for sampling location.
    :type point: Point
    :return: bilinearly interpolated scalar value at given coordinate if (x,y) are within bounds of the scalar field,
     1 otherwise
    """
    if point is not None:
        x = point.x
        y = point.y

    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        metainfo = BilinearSamplingMetaInfo()
        return 1, metainfo

    point = Point(x, y)

    base_point = Point(math.floor(point.x), math.floor(point.y))
    ratios = point - base_point
    inverse_ratios = Point(1.0, 1.0) - ratios

    value00 = sample_at(field, point=base_point)
    value01 = sample_at(field, point=base_point + Point(0, 1))
    value10 = sample_at(field, point=base_point + Point(1, 0))
    value11 = sample_at(field, point=base_point + Point(1, 1))

    interpolated_value0 = value00 * inverse_ratios.y + value01 * ratios.y
    interpolated_value1 = value10 * inverse_ratios.y + value11 * ratios.y
    interpolated_value = interpolated_value0 * inverse_ratios.x + interpolated_value1 * ratios.x

    metainfo = BilinearSamplingMetaInfo(value00, value01, value10, value11, ratios, inverse_ratios)

    return interpolated_value, metainfo


def bilinear_sample_at_replacement(field, x=0, y=0, point=None, replacement=1):
    """
    Sample from a 2D scalar field at a given coordinate with bilinear interpolation.
    If coordinate is out-of-bounds, uses the replacement argument for all samples that fall out of bounds during
    the interpolation.
    Works with either a named point argument or x and y coordinates positionally following the filed argument.
    In case all three are specified and point is not None, it will override the x and y arguments.
    :param replacement: value to use as replacement when sampling out-of-bounds
    :param field: field from which to sample
    :type field: numpy.ndarray
    :param x: x coordinate for sampling location
    :type x: int
    :param y: y coordinate for sampling location
    :type y: int
    :param point: full coordinate for sampling location.
    :type point: Point
    :return: bilinearly interpolated scalar value at given coordinate if (x,y) are within bounds of the scalar field,
     1 otherwise
    """
    if point is not None:
        x = point.x
        y = point.y

    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        return 1

    point = Point(x, y)

    base_point = Point(math.floor(point.x), math.floor(point.y))
    ratios = point - base_point
    inverse_ratios = Point(1.0, 1.0) - ratios

    value00 = sample_at_replacement(field, replacement, point=base_point)
    value01 = sample_at_replacement(field, replacement, point=base_point + Point(0, 1))
    value10 = sample_at_replacement(field, replacement, point=base_point + Point(1, 0))
    value11 = sample_at_replacement(field, replacement, point=base_point + Point(1, 1))

    interpolated_value0 = value00 * inverse_ratios.y + value01 * ratios.y
    interpolated_value1 = value10 * inverse_ratios.y + value11 * ratios.y
    interpolated_value = interpolated_value0 * inverse_ratios.x + interpolated_value1 * ratios.x

    return interpolated_value


def bilinear_sample_at_replacement_metainfo(field, x=0, y=0, point=None, replacement=1):
    """
    Sample from a 2D scalar field at a given coordinate with bilinear interpolation.
    If coordinate is out-of-bounds, uses the replacement argument for all samples that fall out of bounds during
    the interpolation.
    Works with either a named point argument or x and y coordinates positionally following the filed argument.
    In case all three are specified and point is not None, it will override the x and y arguments.
    :param replacement: value to use as replacement when sampling out-of-bounds
    :param field: field from which to sample
    :type field: numpy.ndarray
    :param x: x coordinate for sampling location
    :type x: int
    :param y: y coordinate for sampling location
    :type y: int
    :param point: full coordinate for sampling location.
    :type point: Point
    :return: bilinearly interpolated scalar value at given coordinate if (x,y) are within bounds of the scalar field,
     1 otherwise
    """
    if point is not None:
        x = point.x
        y = point.y

    if x < 0 or x >= field.shape[1] or y < 0 or y >= field.shape[0]:
        return 1

    point = Point(x, y)

    base_point = Point(math.floor(point.x), math.floor(point.y))
    ratios = point - base_point
    inverse_ratios = Point(1.0, 1.0) - ratios

    value00 = sample_at_replacement(field, replacement, point=base_point)
    value01 = sample_at_replacement(field, replacement, point=base_point + Point(0, 1))
    value10 = sample_at_replacement(field, replacement, point=base_point + Point(1, 0))
    value11 = sample_at_replacement(field, replacement, point=base_point + Point(1, 1))

    interpolated_value0 = value00 * inverse_ratios.y + value01 * ratios.y
    interpolated_value1 = value10 * inverse_ratios.y + value11 * ratios.y
    interpolated_value = interpolated_value0 * inverse_ratios.x + interpolated_value1 * ratios.x

    metainfo = BilinearSamplingMetaInfo(value00, value01, value10, value11, ratios, inverse_ratios)

    return interpolated_value, metainfo
