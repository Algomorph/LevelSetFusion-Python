#  ================================================================
#  Created by Gregory Kramida on 1/17/19.
#  Copyright (c) 2019 Gregory Kramida
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
import math


class Circle(object):
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class Ray(object):
    def __init__(self, start, direction):
        self.start = start
        self.direction = direction
        if abs(self.direction.dot(self.direction) - 1.0) > 10e-6:
            raise ValueError("direction must be a unit vector")

    def point_along_ray(self, distance_from_start):
        return self.start + distance_from_start * self.direction


def distances_of_ray_intersections_with_circle(circle, ray):
    """
    :type circle Circle
    :param circle:
    :type ray Ray
    :param ray:
    :return:
    """
    # vector from ray origin to circle center:
    v = ray.start - circle.center
    dir_dot_v = ray.direction.dot(v)
    # find distances using quadratic formula:
    under_square_root = dir_dot_v ** 2 - v.dot(v) + circle.radius**2
    if under_square_root < 0.0:
        return []
    elif under_square_root == 0:
        return [-dir_dot_v]
    else:
        square_root = math.sqrt(under_square_root)
        return [-dir_dot_v - square_root, -dir_dot_v + square_root]
