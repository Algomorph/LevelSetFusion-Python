# code for generating TSDF from raster depth images or from hard-coded piecewise-linear functions

# stdlib
import math
from enum import Enum

# libraries
import numpy as np

# local
from utils.point2d import Point2d
import utils.sampling as sampling
import tsdf.ewa as ewa

IGNORE_OPENCV = False

try:
    import cv2
except ImportError:
    IGNORE_OPENCV = True


class DepthInterpolationMethod:
    NONE = 0
    BILINEAR_IMAGE_SPACE = 1
    BILINEAR_TSDF_SPACE = 2
    EWA = 3
    EWA_CPP = 4


def generate_2d_tsdf_field_from_depth_image_bilinear_tsdf_space(depth_image, camera, image_y_coordinate,
                                                                camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                                field_size=128, default_value=1, voxel_size=0.004,
                                                                array_offset=np.array([-64, -64, 64]),
                                                                narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    if default_value == 1:
        field = np.ones((field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    projection_matrix = camera.intrinsics.intrinsic_matrix
    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    y_voxel = 0.0
    w_voxel = 1.0

    for y_field in range(field_size):
        for x_field in range(field_size):
            x_voxel = (x_field + array_offset[0]) * voxel_size
            z_voxel = (y_field + array_offset[2]) * voxel_size  # acts as "Z" coordinate

            point = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
            point_in_camera_space = camera_extrinsic_matrix.dot(point).flatten()

            if point_in_camera_space[2] <= 0:
                continue

            image_x_coordinate = projection_matrix[0, 0] * point_in_camera_space[0] / point_in_camera_space[2] + \
                                 projection_matrix[0, 2]

            ratios = sampling.get_bilinear_ratios(image_x_coordinate, image_y_coordinate)
            base_x = math.floor(image_x_coordinate)
            base_y = math.floor(image_y_coordinate)
            points = [(base_x, base_y), (base_x, base_y + 1), (base_x + 1, base_y), (base_x + 1, base_y + 1)]
            values = []
            for x_image, y_image in points:
                if x_image < 0 or x_image >= depth_image.shape[1]:
                    values.append(1.0)
                    continue
                depth = depth_image[y_image, x_image] * depth_ratio

                signed_distance_to_voxel_along_camera_ray = depth - point_in_camera_space[2]
                if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width:
                    values.append(-1.0)
                elif signed_distance_to_voxel_along_camera_ray > narrow_band_half_width:
                    values.append(1.0)
                else:
                    values.append(signed_distance_to_voxel_along_camera_ray / narrow_band_half_width)
            field[y_field, x_field] = sampling.interpolate_bilinearly(values, ratios)

    return field


def generate_2d_tsdf_field_from_depth_image_bilinear_image_space(depth_image, camera, image_y_coordinate,
                                                                 camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                                 field_size=128, default_value=1, voxel_size=0.004,
                                                                 array_offset=np.array([-64, -64, 64]),
                                                                 narrow_band_width_voxels=20,
                                                                 back_cutoff_voxels=np.inf):
    if default_value == 1:
        field = np.ones((field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    projection_matrix = camera.intrinsics.intrinsic_matrix
    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    y_voxel = 0.0
    w_voxel = 1.0

    for y_field in range(field_size):
        for x_field in range(field_size):
            x_voxel = (x_field + array_offset[0]) * voxel_size
            z_voxel = (y_field + array_offset[2]) * voxel_size  # acts as "Z" coordinate

            point = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
            point_in_camera_space = camera_extrinsic_matrix.dot(point).flatten()

            if point_in_camera_space[2] <= 0:
                continue

            image_x_coordinate = projection_matrix[0, 0] * point_in_camera_space[0] / point_in_camera_space[2] + \
                                 projection_matrix[0, 2]

            if image_x_coordinate < 0 or image_x_coordinate >= depth_image.shape[1]:
                continue

            depth = sampling.bilinear_sample_at(depth_image, image_x_coordinate, image_y_coordinate) * depth_ratio

            if depth <= 0.0:
                continue

            signed_distance_to_voxel_along_camera_ray = depth - point_in_camera_space[2]
            if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width:
                field[y_field, x_field] = -1.0
            elif signed_distance_to_voxel_along_camera_ray > narrow_band_half_width:
                field[y_field, x_field] = 1.0
            else:
                field[y_field, x_field] = signed_distance_to_voxel_along_camera_ray / narrow_band_half_width

    return field


def generate_2d_tsdf_field_from_depth_image_no_interpolation(depth_image, camera, image_y_coordinate,
                                                             camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                                             field_size=128, default_value=1, voxel_size=0.004,
                                                             array_offset=np.array([-64, -64, 64]),
                                                             narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    """
    Assumes camera is at array_offset voxels relative to sdf grid
    :param narrow_band_width_voxels:
    :param array_offset:
    :param camera_extrinsic_matrix: matrix representing transformation of the camera (incl. rotation and translation)
    [ R | T]
    [ 0 | 1]
    :param voxel_size: voxel size, in meters
    :param default_value: default initial TSDF value
    :param field_size:
    :param depth_image:
    :type depth_image: np.ndarray
    :param camera:
    :type camera: calib.camera.DepthCamera
    :param image_y_coordinate:
    :type image_y_coordinate: int
    :return:
    """
    # TODO: use back_cutoff_voxels for additional limit on
    # "if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width" (maybe?)

    if default_value == 1:
        field = np.ones((field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    projection_matrix = camera.intrinsics.intrinsic_matrix
    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    y_voxel = 0.0
    w_voxel = 1.0

    for y_field in range(field_size):
        for x_field in range(field_size):
            x_voxel = (x_field + array_offset[0]) * voxel_size
            z_voxel = (y_field + array_offset[2]) * voxel_size  # acts as "Z" coordinate

            point = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T

            point_in_camera_space = camera_extrinsic_matrix.dot(point).flatten()

            if point_in_camera_space[2] <= 0:
                continue

            image_x_coordinate = int(
                projection_matrix[0, 0] * point_in_camera_space[0] / point_in_camera_space[2]
                + projection_matrix[0, 2] + 0.5)

            if depth_image.ndim > 1:
                if image_x_coordinate < 0 or image_x_coordinate >= depth_image.shape[1]:
                    continue

                depth = depth_image[image_y_coordinate, image_x_coordinate] * depth_ratio

            else:
                if image_x_coordinate < 0 or image_x_coordinate >= depth_image.shape[0]:
                    continue

                depth = depth_image[image_x_coordinate] * depth_ratio

            if depth <= 0.0:
                continue

            signed_distance_to_voxel_along_camera_ray = depth - point_in_camera_space[2]

            if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width:
                field[y_field, x_field] = -1.0
            elif signed_distance_to_voxel_along_camera_ray > narrow_band_half_width:
                field[y_field, x_field] = 1.0
            else:
                field[y_field, x_field] = signed_distance_to_voxel_along_camera_ray / narrow_band_half_width

    return field


tsdf_from_depth_image_generation_functions_2d = {
    DepthInterpolationMethod.NONE: generate_2d_tsdf_field_from_depth_image_no_interpolation,
    DepthInterpolationMethod.BILINEAR_IMAGE_SPACE: generate_2d_tsdf_field_from_depth_image_bilinear_image_space,
    DepthInterpolationMethod.BILINEAR_TSDF_SPACE: generate_2d_tsdf_field_from_depth_image_bilinear_tsdf_space,
    DepthInterpolationMethod.EWA: ewa.generate_2d_tsdf_field_from_depth_image_ewa,
    DepthInterpolationMethod.EWA_CPP: ewa.generate_2d_tsdf_field_from_depth_image_ewa_cpp
}


def generate_2d_tsdf_field_from_depth_image(depth_image, camera, image_y_coordinate,
                                            camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                            field_size=128, default_value=1, voxel_size=0.004,
                                            array_offset=np.array([-64, -64, 64]),
                                            narrow_band_width_voxels=20, back_cutoff_voxels=np.inf,
                                            depth_interpolation_method=DepthInterpolationMethod.NONE):
    return tsdf_from_depth_image_generation_functions_2d[depth_interpolation_method](
        depth_image, camera, image_y_coordinate, camera_extrinsic_matrix, field_size, default_value,
        voxel_size, array_offset, narrow_band_width_voxels, back_cutoff_voxels)


def add_surface_to_2d_tsdf_field_sample(field, consecutive_surface_points, narrow_band_width_voxels=20,
                                        back_cutoff_voxels=np.inf):
    half_width = narrow_band_width_voxels // 2

    for i_point in range(len(consecutive_surface_points) - 1):
        point_a = consecutive_surface_points[i_point]
        point_b = consecutive_surface_points[i_point + 1]
        point = Point2d()
        x_dist = point_b.x - point_a.x
        for x_voxel in range(int(point_a.x), int(point_b.x)):
            point.x = x_voxel
            ratio = (x_voxel - point_a.x) / x_dist
            point.y = point_a.y * (1.0 - ratio) + point_b.y * ratio
            start_point = int(point.y - half_width)
            end_point = int(point.y + min(half_width, back_cutoff_voxels) + 1)
            if point.y - narrow_band_width_voxels < 0:
                raise ValueError("Surface is too close to 0 in the y dimension for a full narrow band representation")
            for y_voxel in range(0, start_point):
                field[y_voxel, x_voxel] = 1.0

            for y_voxel in range(start_point, end_point):
                distance = min(max((point.y - y_voxel) / half_width, -1.0), 1.0)
                field[y_voxel, x_voxel] = distance

            # fill the rest with -1.0 if there is no back_cutoff specified
            if end_point < field.shape[0] and end_point < back_cutoff_voxels:
                field[end_point:, x_voxel] = -1.0
    return field


# use back_cutoff=3 for canonical to replicate effects of the eta parameter in SobolevFusion Sec 3.1
def generate_sample_orthographic_2d_tsdf_field(consecutive_surface_points, size, narrow_band_width_voxels=20,
                                               back_cutoff_voxels=np.inf, default_value=1):
    if default_value == 1:
        field = np.ones((size, size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((size, size), dtype=np.float32)
    else:
        field = np.ndarray((size, size), dtype=np.float32)
        field.fill(default_value)
    return add_surface_to_2d_tsdf_field_sample(field, consecutive_surface_points, narrow_band_width_voxels,
                                               back_cutoff_voxels)


def generate_initial_orthographic_2d_tsdf_fields(field_size=128, narrow_band_width_voxels=20, mimic_eta=False,
                                                 live_smoothing_kernel_size=0,
                                                 canonical_smoothing_kernel_size=0, default_value=1):
    # for simplicity, the coordinates for polygonal surface boundary are specified as integers
    # it is unrealistic to expect true boundary to hit voxels dead-on like this,
    # so I add a small vertical offset to each point
    offset = -0.23
    surface_point_coordinates = np.array([[9, 56],
                                          [14, 66],
                                          [23, 72],
                                          [35, 72],
                                          [44, 65],
                                          [54, 60],
                                          [63, 60],
                                          [69, 64],
                                          [76, 71],
                                          [84, 73],
                                          [91, 72],
                                          [106, 63],
                                          [109, 57]], dtype=np.float32)

    surface_points_extra = np.array([[32, 65],
                                     [36, 65],
                                     [41, 61]], dtype=np.float32)

    live_surface_points = []

    for i_point in range(surface_point_coordinates.shape[0]):
        live_surface_points.append(
            Point2d(surface_point_coordinates[i_point, 0], surface_point_coordinates[i_point, 1]))

    live_extra_points = [Point2d(surf_point_coordinates[0], surf_point_coordinates[1])
                         for surf_point_coordinates in surface_points_extra]

    for live_point in live_surface_points:
        live_point.y += offset  # unrealistic to expect even values
    for live_point in live_extra_points:
        live_point.y += offset

    live_field = generate_sample_orthographic_2d_tsdf_field(live_surface_points, field_size,
                                                            narrow_band_width_voxels=narrow_band_width_voxels,
                                                            default_value=default_value)
    live_field = add_surface_to_2d_tsdf_field_sample(live_field, live_extra_points,
                                                     narrow_band_width_voxels=narrow_band_width_voxels)

    # generate canonical field as live field shifted with a constant offset
    canonical_surface_points = [Point2d(live_point.x, live_point.y + 5.0) for live_point in live_surface_points]
    canonical_extra_points = [Point2d(live_point.x, live_point.y + 5.0) for live_point in live_extra_points]

    if mimic_eta:
        back_cutoff_voxels = 3
    else:
        back_cutoff_voxels = np.inf

    canonical_field = generate_sample_orthographic_2d_tsdf_field(canonical_surface_points, field_size,
                                                                 narrow_band_width_voxels=narrow_band_width_voxels,
                                                                 back_cutoff_voxels=back_cutoff_voxels,
                                                                 default_value=default_value)
    canonical_field = add_surface_to_2d_tsdf_field_sample(canonical_field, canonical_extra_points,
                                                          narrow_band_width_voxels=narrow_band_width_voxels,
                                                          back_cutoff_voxels=back_cutoff_voxels)

    # smooth live & canonical as necessary
    if live_smoothing_kernel_size > 0 and not IGNORE_OPENCV:
        live_field = cv2.GaussianBlur(live_field, (live_smoothing_kernel_size, live_smoothing_kernel_size), 0,
                                      borderType=cv2.BORDER_REPLICATE)
    if canonical_smoothing_kernel_size > 0 and not IGNORE_OPENCV:
        canonical_field = cv2.GaussianBlur(canonical_field,
                                           (canonical_smoothing_kernel_size, canonical_smoothing_kernel_size), 0,
                                           borderType=cv2.BORDER_REPLICATE)

    return live_field, canonical_field


def generate_3d_tsdf_field_from_depth_image(depth_image, camera,
                                            camera_extrinsic_matrix=np.eye(4, dtype=np.float32),
                                            field_size=128, default_value=1, voxel_size=0.004,
                                            array_offset=np.array([-64, -64, 64]),
                                            narrow_band_width_voxels=20, back_cutoff_voxels=np.inf):
    """
    Assumes camera is at array_offset voxels relative to sdf grid
    :param narrow_band_width_voxels:
    :param array_offset:
    :param camera_extrinsic_matrix: matrix representing transformation of the camera (incl. rotation and translation)
    [ R | T]
    [ 0 | 1]
    :param voxel_size: voxel size, in meters
    :param default_value: default initial TSDF value
    :param field_size:
    :param depth_image:
    :type depth_image: np.ndarray
    :param camera:
    :type camera: calib.camera.DepthCamera
    :param image_y_coordinate:
    :type image_y_coordinate: int
    :return:
    """
    # TODO: use back_cutoff_voxels for additional limit on
    # "if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width" (maybe?)

    if default_value == 1:
        field = np.ones((field_size, field_size, field_size), dtype=np.float32)
    elif default_value == 0:
        field = np.zeros((field_size, field_size, field_size), dtype=np.float32)
    else:
        field = np.ndarray((field_size, field_size, field_size), dtype=np.float32)
        field.fill(default_value)

    projection_matrix = camera.intrinsics.intrinsic_matrix

    depth_ratio = camera.depth_unit_ratio
    narrow_band_half_width = narrow_band_width_voxels / 2 * voxel_size  # in metric units

    w_voxel = 1.0

    for z_field in range(field_size):
        for y_field in range(field_size):
            for x_field in range(field_size):

                x_voxel = (x_field + array_offset[0]) * voxel_size
                y_voxel = (y_field + array_offset[1]) * voxel_size
                z_voxel = (z_field + array_offset[2]) * voxel_size

                point = np.array([[x_voxel, y_voxel, z_voxel, w_voxel]], dtype=np.float32).T
                point_in_camera_space = camera_extrinsic_matrix.dot(point).flatten()

                if point_in_camera_space[2] <= 0:
                    continue

                image_x_coordinate = int(
                    projection_matrix[0, 0] * point_in_camera_space[0] / point_in_camera_space[2]
                    + projection_matrix[0, 2] + 0.5)
                image_y_coordinate = int(
                    projection_matrix[1, 1] * point_in_camera_space[1] / point_in_camera_space[2]
                    + projection_matrix[1, 2] + 0.5
                )

                if image_x_coordinate < 0 or image_x_coordinate >= depth_image.shape[1] \
                        or image_y_coordinate < 0 or image_y_coordinate >= depth_image.shape[0]:
                    continue

                depth = depth_image[image_y_coordinate, image_x_coordinate] * depth_ratio

                if depth <= 0.0:
                    continue

                signed_distance_to_voxel_along_camera_ray = depth - point_in_camera_space[2]

                if signed_distance_to_voxel_along_camera_ray < -narrow_band_half_width:
                    field[y_field, x_field] = -1.0
                elif signed_distance_to_voxel_along_camera_ray > narrow_band_half_width:
                    field[y_field, x_field] = 1.0
                else:
                    field[y_field, x_field] = signed_distance_to_voxel_along_camera_ray / narrow_band_half_width

    return field
