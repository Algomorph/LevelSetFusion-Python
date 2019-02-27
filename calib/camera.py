"""
/home/algomorph/Factory/calib_video_opencv/intrinsics/video.py.
Created on Mar 21, 2016.
@author: Gregory Kramida
@licence: Apache v2

Copyright 2016 Gregory Kramida

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# libraries
from lxml import etree
import numpy as np
from cv2 import remap, INTER_LINEAR

# local
import calib.utils.xml_io as xml

from calib.utils.custom_format_io import parse_line_as_float_tuple, parse_lines_as_matrix

DEFAULT_RESOLUTION = (1080, 1920)


def _resolution_from_xml(element):
    resolution_elem = element.find("resolution")
    width = int(resolution_elem.find("width").text)
    height = int(resolution_elem.find("height").text)
    return height, width


def _resolution_to_xml(element, resolution):
    resolution_elem = etree.SubElement(element, "resolution")
    width_elem = etree.SubElement(resolution_elem, "width")
    width_elem.text = str(resolution[1])
    height_elem = etree.SubElement(resolution_elem, "height")
    height_elem.text = str(resolution[0])


def _meta_info_from_xml(element):
    error = float(element.find("error").text)
    time = float(element.find("time").text)
    if element.find("calibration_image_count") is not None:
        calibration_image_count = int(element.find("calibration_image_count").text)
    else:
        calibration_image_count = 0
    return error, time, calibration_image_count


def _meta_info_to_xml(element, error, time, calibration_image_count):
    error_element = etree.SubElement(element, "error")
    error_element.text = str(error)
    time_element = etree.SubElement(element, "time")
    time_element.text = str(time)
    calibration_image_count_element = etree.SubElement(element, "calibration_image_count")
    calibration_image_count_element.text = str(calibration_image_count)


class Camera(object):
    """
    Represents a video object & camera that was used to capture it, a wrapper around OpenCV's video_capture
    """

    class Intrinsics(object):
        """
        Represents videos of a camera, i.e. intrinsic matrix & distortion coefficients
        """

        def __init__(self, resolution, intrinsic_matrix=None,
                     distortion_coeffs=None,
                     error=-1.0, time=0.0, calibration_image_count=0):
            """
            Constructor
            @type intrinsic_matrix: numpy.ndarray
            @param intrinsic_matrix: intrinsic matrix (3x3)
            @type distortion_coeffs: numpy.ndarray
            @param distortion_coeffs: distortion coefficients (1x8)
            @type resolution: tuple[int]
            @param resolution: pixel resolution (height,width) of the camera
            @type error: float
            @param error: mean square distance error to object points after reprojection
            @type  time: float
            @param time: calibration time in seconds
            """
            if intrinsic_matrix is None:
                intrinsic_matrix = np.eye(3, dtype=np.float64)
                intrinsic_matrix[0, 2] = resolution[1] / 2
                intrinsic_matrix[1, 2] = resolution[0] / 2
            self.intrinsic_matrix = intrinsic_matrix
            if distortion_coeffs is None:
                distortion_coeffs = np.zeros((8, 1), np.float64)
            self.distortion_coeffs = distortion_coeffs
            self.resolution = resolution
            self.error = error
            self.time = time
            self.timestamp = None
            self.calibration_image_count = calibration_image_count

        def to_xml(self, root_element, as_sequence=False):
            """
            Build an xml node representation of this object under the provided root xml element
            @type root_element:  lxml.etree.SubElement
            @param root_element: the root element to build under
            @type as_sequence: bool
            @param as_sequence: whether to generate XML for sequences (see OpenCV's documentation on XML/YAML persistence)
            """
            if not as_sequence:
                elem_name = self.__class__.__name__
            else:
                elem_name = "_"
            intrinsics_elem = etree.SubElement(root_element, elem_name)
            _resolution_to_xml(intrinsics_elem, self.resolution)
            xml.make_opencv_matrix_xml_element(intrinsics_elem, self.intrinsic_matrix, "intrinsic_mat")
            xml.make_opencv_matrix_xml_element(intrinsics_elem, self.distortion_coeffs, "distortion_coeffs")
            _meta_info_to_xml(intrinsics_elem, self.error, self.time, self.calibration_image_count)

        def __str__(self):
            return (("{:s}\nResolution (h,w): {:s}\n" +
                     "Intrinsic matrix:\n{:s}\nDistortion coefficients:\n{:s}\n" +
                     "Error: {:f}\nTime: {:f}\nImage count: {:d}")
                    .format(self.__class__.__name__, str(self.resolution), str(self.intrinsic_matrix),
                            str(self.distortion_coeffs), self.error, self.time, self.calibration_image_count))

        @staticmethod
        def from_xml(element):
            """
            @type element: lxml.etree.SubElement
            @param element: the element to construct an CameraIntrinsics object from
            @return a new CameraIntrinsics object constructed from XML node with matrices in OpenCV format
            """
            if element is None:
                return Camera.Intrinsics(DEFAULT_RESOLUTION)
            resolution = _resolution_from_xml(element)
            intrinsic_mat = xml.parse_xml_matrix(element.find("intrinsic_mat"))
            distortion_coeffs = xml.parse_xml_matrix(element.find("distortion_coeffs"))
            error, time, calibration_image_count = _meta_info_from_xml(element)
            return Camera.Intrinsics(resolution, intrinsic_mat, distortion_coeffs, error, time, calibration_image_count)

    # TODO: Rename to and combine with "Pose" from calib.geom

    class Extrinsics(object):
        def __init__(self, rotation=None, translation=None, error=-1.0, time=0.0, calibration_image_count=0,
                     extrinsics_matrix=None):
            """
            Constructor
            @type rotation: numpy.ndarray
            @param rotation: 3x3 rotation matrix from camera 0 to camera 1
            @type translation: numpy.ndarray
            @param translation: a 3x1 translation vector from camera 0 to camera 1
            @type error: float
            @param error: mean square distance error to object points after reprojection
            @type  time: float
            @param time: calibration time in seconds
            """
            if extrinsics_matrix is not None:
                rotation = extrinsics_matrix[0:3, 0:3]
                translation = extrinsics_matrix[0:3, 3].reshape(1, 3)
            if rotation is None:
                rotation = np.eye(3, dtype=np.float64)
            if translation is None:
                translation = np.zeros((1, 3), np.float64)
            self.rotation = rotation
            self.translation = translation
            self.error = error
            self.time = time
            self.timestamp = None
            self.calibration_image_count = calibration_image_count

        def __str__(self):
            return ("{:s}\nRotation:\n{:s}\nTranslation:\n{:s}\nError: {:f}\nTime: {:f}\nImage count: {:d}"
                    .format(self.__class__.__name__, str(self.rotation),
                            str(self.translation), self.error, self.time, self.calibration_image_count))

        def to_xml(self, root_element, as_sequence=False):
            """
            Build an xml node representation of this object under the provided root xml element
            @type root_element:  lxml.etree.SubElement
            @param root_element: the root element to build under
            @type as_sequence: bool
            @param as_sequence: whether to generate XML for sequences
            (see OpenCV's documentation on XML/YAML persistence)

            """
            if not as_sequence:
                elem_name = self.__class__.__name__
            else:
                elem_name = "_"

            extrinsics_elem = etree.SubElement(root_element, elem_name)

            xml.make_opencv_matrix_xml_element(extrinsics_elem, self.rotation, "rotation")
            xml.make_opencv_matrix_xml_element(extrinsics_elem, self.translation, "translation")
            _meta_info_to_xml(extrinsics_elem, self.error, self.time, self.calibration_image_count)

        @staticmethod
        def from_xml(element):
            """
            Build a CameraExtrinsics object out of the provided XML node with matrices in
            OpenCV format
            @type element: lxml.etree.SubElement
            @param element: the element to construct an StereoRig object from
            @rtype: calib.CameraExtrinsics|None
            @return a new StereoRig object constructed from given XML node, None if element is None
            """
            if element is None:
                return Camera.Extrinsics()
            rotation = xml.parse_xml_matrix(element.find("rotation"))
            translation = xml.parse_xml_matrix(element.find("translation"))
            error, time, calibration_image_count = _meta_info_from_xml(element)
            return Camera.Extrinsics(rotation, translation, error, time, calibration_image_count)

    def __init__(self, resolution=None, intrinsics=None, extrinsics=None):
        """
        Build a camera with the specified parameters
        """
        if resolution is None:
            resolution = DEFAULT_RESOLUTION
            if intrinsics is not None and intrinsics.resolution is not None:
                resolution = intrinsics.resolution
        if intrinsics is None:
            self.intrinsics = Camera.Intrinsics(resolution)
        else:
            self.intrinsics = intrinsics
        if extrinsics is None:
            self.extrinsics = Camera.Extrinsics()
        else:
            self.extrinsics = extrinsics

        # for undistortion
        self.map_x = None
        self.map_y = None

    def rectify_image(self, image):
        return remap(image, self.map_x, self.map_y, INTER_LINEAR)

    def copy(self):
        return Camera(intrinsics=self.intrinsics, extrinsics=self.extrinsics)

    def __str__(self, *args, **kwargs):
        if self.extrinsics.error > 0.0:
            extrinsics_string = "\n" + str(self.extrinsics)
        else:
            extrinsics_string = ""
        return Camera.__name__ + "\n" + str(self.intrinsics) + extrinsics_string

    def to_xml(self, root_element, as_sequence=False):
        """
        Build an xml node representation of this object under the provided root xml element
        @type as_sequence: bool
        @param as_sequence: use sequence opencv XML notation, i.e. XML element name set to "_"
        @type root_element:  lxml.etree.SubElement
        @param root_element: the root element to build under
        """
        if not as_sequence:
            element_tag = self.__class__.__name__
        else:
            element_tag = "_"
        camera_elem = etree.SubElement(root_element, element_tag)
        if self.intrinsics:
            self.intrinsics.to_xml(camera_elem, False)
        if self.extrinsics and self.extrinsics.error > 0.0:
            self.extrinsics.to_xml(camera_elem, False)
        return camera_elem

    @staticmethod
    def from_xml(element):
        """
        @type element: lxml.etree.SubElement
        @param element: the element to construct an CameraIntrinsics object from
        @return a new Camera object constructed from XML node with matrices in OpenCV format
        """
        intrinsics_elem = element.find(Camera.Intrinsics.__name__)
        if intrinsics_elem is None:
            # legacy format
            intrinsics_elem = element.find("CameraIntrinsics")
        if intrinsics_elem is not None:
            intrinsics = Camera.Intrinsics.from_xml(intrinsics_elem)
        else:
            intrinsics = Camera.Intrinsics(DEFAULT_RESOLUTION)
        extrinsics_elem = element.find(Camera.Extrinsics.__name__)
        if extrinsics_elem is not None:
            extrinsics = Camera.Extrinsics.from_xml(extrinsics_elem)
        else:
            extrinsics = Camera.Extrinsics()
        return Camera(None, intrinsics, extrinsics)

    @staticmethod
    def from_infinitam_format(resolution_line, intrinsics_lines):
        resolution = parse_line_as_float_tuple(resolution_line)
        intrinsics_matrix = parse_lines_as_matrix(intrinsics_lines)
        intrinsics = Camera.Intrinsics(resolution, intrinsic_matrix=intrinsics_matrix)
        return Camera(resolution, intrinsics)


class DepthCamera(Camera):
    def __init__(self, resolution=None, intrinsics=None, extrinsics=None, depth_unit_ratio=0.001, camera=None):
        super(DepthCamera, self).__init__(resolution, intrinsics, extrinsics)
        if camera is not None:
            super(DepthCamera, self).__init__(camera.resolution, camera.intrinsics, camera.extrinsics)
        self.depth_unit_ratio = depth_unit_ratio

    def to_xml(self, root_element, as_sequence=False):
        if not as_sequence:
            element_tag = self.__class__.__name__
        else:
            element_tag = "_"
        camera_element = super(DepthCamera, self).to_xml(root_element, as_sequence)
        camera_element.tag = element_tag
        depth_unit_ratio_element = etree.SubElement(camera_element, "DepthUnitRatio")
        depth_unit_ratio_element.text = str(self.depth_unit_ratio)

    def __str__(self, *args, **kwargs):
        # TODO: need a more explicit way to say if the extrinsics have been set.
        # What if extrinsics were manually entered with "0" error?
        if self.extrinsics.error > 0.0:
            extrinsics_string = "\n" + str(self.extrinsics)
        else:
            extrinsics_string = ""
        return DepthCamera.__name__ + "\n" + str(
            self.intrinsics) + extrinsics_string + "\nDepthUnitRatio: {:f}".format(self.depth_unit_ratio)

    @staticmethod
    def from_xml(element):
        depth_unit_ratio = float(element.find("DepthUnitRatio").text)
        return DepthCamera(depth_unit_ratio=depth_unit_ratio, camera=Camera.from_xml(element))

    @staticmethod
    def from_infinitam_format(resolution_line, intrinsics_lines, extrinsics_lines, depth_parameters_line):
        """
        Generate a DepthCamera object from specific lines in the InfiniTAM-formatted calibration text file
        :param resolution_line: line containing the resolution
        :param intrinsics_lines: lines containing the intrinsic matrix
        :param extrinsics_lines: lines containing the extrinsic matrix
        :param depth_parameters_line: depth parameters
        :return:
        """
        resolution = parse_line_as_float_tuple(resolution_line)
        intrinsics_params = parse_lines_as_matrix(intrinsics_lines)
        intrinsics_matrix = np.eye(3, dtype=np.float64)
        intrinsics_matrix[0, 0] = intrinsics_params[0, 0]
        intrinsics_matrix[1, 1] = intrinsics_params[0, 1]
        intrinsics_matrix[0, 2] = intrinsics_params[1, 0]
        intrinsics_matrix[1, 2] = intrinsics_params[1, 1]

        intrinsics = Camera.Intrinsics(resolution, intrinsic_matrix=intrinsics_matrix)
        extrinsics_matrix = parse_lines_as_matrix(extrinsics_lines)
        extrinsics = Camera.Extrinsics(extrinsics_matrix=extrinsics_matrix)
        tokens = depth_parameters_line.strip().split(" ")
        if tokens[0] != "kinect" or tokens[0] != "affine":
            ratio_token = tokens[1]
        else:
            ratio_token = tokens[0]
        depth_unit_ratio = float(ratio_token)
        if depth_unit_ratio == 0.0:
            depth_unit_ratio = 0.001
        return DepthCamera(resolution=resolution, intrinsics=intrinsics, extrinsics=extrinsics,
                           depth_unit_ratio=depth_unit_ratio)
