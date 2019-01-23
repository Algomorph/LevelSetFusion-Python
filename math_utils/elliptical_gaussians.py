#  ================================================================
#  Created by Gregory Kramida on 1/22/19.
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

import numpy as np
import math
import cv2


class ImplicitEllipse:
    def __init__(self, A=None, B=None, C=None, F=1, Q=None):
        if Q is not None:
            self.conic_matrix = self.Q = Q
            self.A = Q[0, 0]
            self.B = Q[0, 1] * 2
            self.C = Q[1, 1]
            if Q[0, 1] != Q[1, 0]:
                raise ValueError("Conic matrix should be symmetric! Given: {:s}".format(str(Q)))
        else:
            self.A = A
            self.B = B
            self.C = C
            self.conic_matrix = self.Q = np.array([[A, B / 2], [B / 2, C]])
        self.F = F

    def get_bounds(self):
        if abs(float(self.B)) < 10e-6:
            x_max = math.sqrt(self.F / self.A)
            y_max = math.sqrt(self.F / self.C)
        else:
            x_max = math.sqrt(self.F / ((4 * self.A * self.C ** 2) / (self.B ** 2) - self.C))
            y_max = math.sqrt(self.F / ((4 * self.A ** 2 * self.C) / (self.B ** 2) - self.A))

        x_min = -x_max
        y_min = -y_max
        return np.array([[x_min, x_max],
                         [y_min, y_max]])

    def visualize(self, scale=100, margin=5):
        image_bounds = (self.get_bounds() * scale).astype(np.int32)
        # set both dimensions to largest one, to make the picture square
        if image_bounds[0, 1] > image_bounds[1, 1]:
            image_bounds[1, :] = image_bounds[0, :]
        else:
            image_bounds[0, :] = image_bounds[1, :]

        image_bounds[:, 0] -= margin
        image_bounds[:, 1] += margin
        center_offset = image_bounds[:, 1].copy().reshape(-1, 1)
        image_bounds += center_offset
        image_shape = (image_bounds[1, 1], image_bounds[0, 1], 3)
        image = np.ones(image_shape, np.uint8) * 255

        for x_pixel in range(margin, image_shape[1] - margin):
            x_point = float(x_pixel - center_offset[0]) / scale
            a = self.C
            b = self.B * x_point
            c = self.A * x_point ** 2 - self.F
            under_root = b ** 2 - 4 * a * c
            if under_root < 0.0:
                continue
            addand = math.sqrt(under_root)

            denominator = (-b - addand)
            y_0 = denominator / (2 * a) if abs(denominator) > 10e-6 else 0.0
            denominator = (-b + addand)
            y_1 = denominator / (2 * a) if abs(denominator) > 10e-6 else 0.0
            y_pixel_0 = int(y_0 * scale + center_offset[1] + 0.5)
            y_pixel_1 = int(y_1 * scale + center_offset[1] + 0.5)
            image[y_pixel_0, x_pixel] = (0, 0, 0)
            image[y_pixel_1, x_pixel] = (0, 0, 0)

        # to fill it out a bit better
        # (I know the more "proper" way would probably be doing a parametric via angles. This is easier right now)
        for y_pixel in range(margin, image_shape[0] - margin):
            y_point = float(y_pixel - center_offset[1]) / scale
            a = self.A
            b = self.B * y_point
            c = self.C * y_point ** 2 - self.F
            under_root = b ** 2 - 4 * a * c
            if under_root < 0.0:
                continue
            addand = math.sqrt(under_root)

            denominator = (-b - addand)
            x_0 = denominator / (2 * a) if abs(denominator) > 10e-8 else 0.0
            denominator = (-b + addand)
            x_1 = denominator / (2 * a) if abs(denominator) > 10e-8 else 0.0
            x_pixel_0 = int(x_0 * scale + center_offset[1] + 0.5)
            x_pixel_1 = int(x_1 * scale + center_offset[1] + 0.5)
            image[y_pixel, x_pixel_0] = (0, 0, 0)
            image[y_pixel, x_pixel_1] = (0, 0, 0)

        cv2.imshow("ellipse", image)
        cv2.waitKey()

    @staticmethod
    def unit_circle():
        return ImplicitEllipse(Q=np.eye(2), F=1)


def implicit_ellipse_from_radii_and_angle(radius_a, radius_b, angle, F=1.0):
    M = np.array([[radius_a * math.cos(angle), -radius_a * math.sin(angle)],
                  [radius_b * math.sin(angle), radius_b * math.cos(angle)]])
    Q1 = np.eye(2)
    Minv = np.linalg.inv(M)
    Q2 = Minv.dot(Q1).dot(Minv.T)
    return ImplicitEllipse(Q=Q2, F=F)


def implicit_ellipse_from_radii_and_angle2(radius_a, radius_b, angle):
    M = np.array([[radius_a * math.cos(angle), radius_a * math.sin(angle)],
                  [-radius_b * math.sin(angle), radius_b * math.cos(angle)]])
    Q1 = np.eye(2)
    Minv = np.linalg.inv(M)
    Q2 = Minv.dot(Q1).dot(Minv.T)
    F = (radius_a * radius_b) ** 2
    return ImplicitEllipse(Q=Q2, F=F)


class EllipticalGaussian:
    def __init__(self, ellipse):
        self.ellipse = ellipse
        self.V = np.linalg.inv(ellipse)
        self.normalization_factor = 1 / (2 * math.pi * np.linalg.det(self.V) ** (1 / 2))

    def compute(self, point):
        return self.normalization_factor * np.exp((-1 / 2) * point.T.dot(self.ellipse.Q).dot(point))
