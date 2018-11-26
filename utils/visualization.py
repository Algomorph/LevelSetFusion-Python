#  ================================================================
#  Created by Gregory Kramida on 9/11/18.
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
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from utils.sampling import get_focus_coordinates

# TODO: take care of fromstring deprecation issue throughout file

VIEW_SCALING_FACTOR = 8

IGNORE_OPENCV = False

try:
    import cv2
except ImportError:
    IGNORE_OPENCV = True


def process_cv_esc():
    if not IGNORE_OPENCV:
        key = cv2.waitKey()
        if key == 27:
            sys.exit(0)
    else:
        pass


def exit_func(event):
    print("Key pressed:", event.key)
    if event.key == 'escape' or event.key == 'q':
        sys.exit(0)


def rescale_depth_to_8bit(depth_image):
    return ((depth_image.astype(np.float64) - 500) * 255 / 4000).astype(np.uint8)


def highlight_row_on_gray(gray_image, ix_row):
    bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    image_with_highlight = np.zeros_like(bgr)
    image_with_highlight[:, :, 0] = 255
    # wash out the b and r channels
    image_with_highlight[:, :, 1] = bgr[:, :, 1] + 0.25 * (255 - bgr[:, :, 1])
    image_with_highlight[:, :, 2] = bgr[:, :, 2] + 0.75 * (255 - bgr[:, :, 2])
    image_with_highlight[ix_row, :, 0] = bgr[ix_row, :, 0]
    image_with_highlight[ix_row, :, 1] = bgr[ix_row, :, 1]
    image_with_highlight[ix_row, :, 2] = bgr[ix_row, :, 2]
    return image_with_highlight


# TODO: all vizualization functions that currently accept an OpenCV writer and write image should instead simply
# produce an image, which should be written, if necessary, by another routine

def make_vector_field_plot(vector_field_video_writer, warp_field, iteration_number=None, sparsity_factor=1,
                           use_magnitude_for_color=True, scale=1.0, vectors_name="Warp vectors"):
    fig = plt.figure(figsize=(23.6, 14))
    ax = fig.gca()
    if iteration_number is not None:
        ax.set_title("{:s}, iteration {:d}".format(vectors_name, iteration_number))
    else:
        ax.set_title(vectors_name)
    ax.invert_yaxis()

    width = warp_field.shape[0] / sparsity_factor
    height = warp_field.shape[1] / sparsity_factor

    x_grid = np.arange(0, width)
    y_grid = np.arange(0, height)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    u_data = warp_field[::sparsity_factor, ::sparsity_factor, 0]
    v_data = warp_field[::sparsity_factor, ::sparsity_factor, 1]

    if use_magnitude_for_color:
        magnitudes = np.linalg.norm(warp_field, axis=2) / 10.0 * scale
        max_value = 0.5
        magnitudes /= max_value
        magnitudes[magnitudes > 1.0] = 1.0
        # color_map = plt.get_cmap("magma")
        # color = color_map(magnitudes)
        color = magnitudes
    else:
        color = 10 + np.sqrt((y_grid - height / 2.) ** 2 + (x_grid - width / 2.) ** 2)

    # ax.quiver(x_grid, y_grid, u_data, v_data, edgecolor='k', facecolor='None', linewidth=.5, units='dots', width=1)
    ax.quiver(x_grid, y_grid, u_data, v_data, color, edgecolor='k', linewidth=0, units='dots', width=1,
              scale=1.0 / scale, angles='xy', scale_units='xy')
    fig.canvas.draw()

    plt.close(fig)

    plot_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plot_image = plot_image[110:1310, 240:2160]
    bgr = cv2.cvtColor(plot_image, cv2.COLOR_RGB2BGR)
    vector_field_video_writer.write(bgr)


def make_3d_plots(live_video_writer_3d, canonical_field, warped_live_field):
    """
    Makes a 3D plot of the live sdf, with the SDF value plotted along the (vertical) Z axis
    :param live_video_writer_3d:
    :param canonical_field:
    :param warped_live_field:
    :return:
    """
    # plot warped live field
    fig = plt.figure(figsize=(16, 10))
    fig.canvas.mpl_connect('key_press_event', exit_func)
    ax = fig.gca(projection='3d')

    # Make live data.
    x_grid = np.arange(0, warped_live_field.shape[0])
    y_grid = np.arange(0, warped_live_field.shape[1])

    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    # Chop of "unknowns" from both sides
    x_start = 9
    x_end = 109
    x_grid_cropped = x_grid[:, x_start:x_end]
    y_grid_cropped = y_grid[:, x_start:x_end]
    live_z = warped_live_field * 10
    live_z_cropped = live_z[:, x_start:x_end, ]

    # Plot the surface.
    surf = ax.plot_surface(x_grid_cropped, y_grid_cropped, live_z_cropped, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, shade=True, rcount=128, ccount=x_start - x_end)

    canonical_z = canonical_field * 10
    canonical_z_cropped = canonical_z[:, x_start: x_end]

    # colors = cm.viridis(canonical_z_cropped)
    wire = ax.plot_wireframe(x_grid_cropped, y_grid_cropped, canonical_z_cropped, rcount=128, linestyles='solid',
                             linewidth=0.7, ccount=x_start - x_end, color=(0, 0, 0, 0.5))
    # Customize the z axis.
    ax.set_zlim(-10.5, 10.05)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.view_init(20, 30)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.canvas.draw()

    plot_image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plot_image = plot_image[150:870, 200:1430]
    plt.close(fig)
    live_video_writer_3d.write(plot_image)


def warp_field_to_heatmap(warp_field, scale=VIEW_SCALING_FACTOR, use_pixel_labels=True):
    magnitudes = np.linalg.norm(warp_field, axis=2)
    color_map = plt.get_cmap("magma")
    max_value = 0.1
    magnitudes /= max_value
    magnitudes[magnitudes > 1.0] = 1.0
    colors = color_map(magnitudes)
    # use only first three channels
    colors = colors[:, :, :3]
    heatmap = (colors * 255).astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)
    if use_pixel_labels and not IGNORE_OPENCV:
        field_size = warp_field.shape[0]
        image_size = field_size * scale
        for y in range(0, image_size, scale * 2):
            dash_start = scale * 6
            heatmap[y, 0:dash_start] = (255, 255, 255)
            cv2.putText(heatmap, str(y // scale), (2, y + 11), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255))
            for x in range(dash_start, image_size, scale):
                heatmap[y, x] = (255, 255, 255)
        for x in range(scale * 6, image_size, scale * 4):
            dash_start = scale * 3
            heatmap[0:dash_start, x] = (128, 255, 128)
            cv2.putText(heatmap, str(x // scale), (x + 2, 16), cv2.FONT_HERSHEY_PLAIN, 0.9, (128, 255, 128))
            for y in range(dash_start, image_size, scale):
                heatmap[y, x] = (128, 255, 128)
    return heatmap


def sdf_field_to_image(field, scale=8):
    return ((field + 1.0) * 255 / 2.0).astype(np.uint8).repeat(scale, axis=0).repeat(scale, axis=1)


def mark_focus_coordinate_on_sdf_image(image, scale=8):
    focus_coordinates = get_focus_coordinates()
    x_start = focus_coordinates[0] * scale
    y_start = focus_coordinates[1] * scale

    x_end = x_start + scale
    y_end = y_start + scale
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if scale < 3:
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                out[y, x] = (0, 0, 255)
    else:
        for y in range(y_start, y_end):
            out[y, x_start] = (0, 0, 255)
            out[y, x_end - 1] = (0, 0, 255)
        for x in range(x_start + 1, x_end - 1):
            out[y_start, x] = (0, 0, 255)
            out[y_end, x] = (0, 0, 255)
    return out


def logs_ordered_by_pixel_positions_number(neighborhood_log):
    keys = list(neighborhood_log.keys())
    sorted_keys = sorted(keys, key=lambda tup: tup[0] + tup[1] * 10)
    logs = []
    for key in sorted_keys:
        logs.append((neighborhood_log[key], key))
    return logs


def visualize_and_save_sdf_and_warp_magnitude_progression(focus_coordinates, focus_neighborhood_log, out_path):
    plt.clf()
    plt.figure(figsize=(19.2, 11.2))
    plot_number = 1
    logs = logs_ordered_by_pixel_positions_number(focus_neighborhood_log)
    for log, (x, y) in logs:
        plt.subplot(330 + plot_number)
        plt.plot(log.sdf_values, "b")
        plt.plot([0, len(log.sdf_values)], [log.canonical_sdf, log.canonical_sdf], 'k')
        plt.title("{:d},{:d}".format(x, y))
        plt.grid(True)
        plot_number += 1
    plt.suptitle("SDF Value Progression")
    filename = "sdf_values_for_pt_{:d}_{:d}_neighborhood.png".format(focus_coordinates[0], focus_coordinates[1])
    plt.savefig(os.path.join(out_path, filename))
    plt.clf()
    plt.figure(figsize=(19.2, 11.2))
    plot_number = 1
    for log, (x, y) in logs:
        plt.subplot(330 + plot_number)
        plt.plot(log.warp_magnitudes, "g")
        plt.title("{:d},{:d}".format(x, y))
        plot_number += 1
        plt.grid(True)
    plt.suptitle("Warp Magnitude Progression")
    filename = "warp_magnitudes_for_pt_{:d}_{:d}_neighborhood.png".format(focus_coordinates[0],
                                                                          focus_coordinates[1])
    plt.savefig(os.path.join(out_path, filename))
    plt.close('all')


def visualzie_and_save_energy_and_max_warp_progression(log, out_path):
    plt.clf()
    plt.figure(figsize=(18.6, 11.2))
    plt.title("Energies")
    ax = plt.gca()
    x = np.arange(len(log.data_energies))
    labels = ["Data", "Smoothing", "Level Set"]
    ax.stackplot(x, log.data_energies, log.smoothing_energies, log.level_set_energies, labels=labels)
    ax.legend(loc='upper right')
    filename = "energy_stackplot.png"
    plt.savefig(os.path.join(out_path, filename))
    plt.clf()
    plt.figure(figsize=(18.6, 11.2))
    plt.plot(log.max_warps, "k")
    filename = "max_warp_plot.png"
    plt.savefig(os.path.join(out_path, filename))
    plt.clf()
    plt.close('all')


def visualize_and_save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor=8):
    canonical_visualized = sdf_field_to_image(canonical_field, scale=view_scaling_factor)
    canonical_visualized = mark_focus_coordinate_on_sdf_image(canonical_visualized)
    canonical_visualized_unscaled = sdf_field_to_image(canonical_field, scale=1)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    cv2.imshow("canonical SDF", canonical_visualized)
    cv2.imwrite(os.path.join(out_path, 'unscaled_initial_canonical.png'), canonical_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, 'initial_canonical.png'), canonical_visualized)
    process_cv_esc()
    live_visualized = sdf_field_to_image(live_field, scale=view_scaling_factor)
    live_visualized = mark_focus_coordinate_on_sdf_image(live_visualized)
    live_visualized_unscaled = sdf_field_to_image(live_field, scale=1)
    cv2.imwrite(os.path.join(out_path, "unscaled_initial_live.png"), live_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, "initial_live.png"), live_visualized)
    cv2.imshow("live SDF", live_visualized)
    process_cv_esc()
    cv2.destroyAllWindows()


def save_initial_fields(canonical_field, live_field, out_path, view_scaling_factor=8):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    canonical_visualized = sdf_field_to_image(canonical_field, scale=view_scaling_factor)
    canonical_visualized = mark_focus_coordinate_on_sdf_image(canonical_visualized, scale=view_scaling_factor)
    canonical_visualized_unscaled = sdf_field_to_image(canonical_field, scale=1)

    cv2.imwrite(os.path.join(out_path, 'unscaled_initial_canonical.png'), canonical_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, 'initial_canonical.png'), canonical_visualized)

    live_visualized = sdf_field_to_image(live_field, scale=view_scaling_factor)
    live_visualized = mark_focus_coordinate_on_sdf_image(live_visualized, scale=view_scaling_factor)
    live_visualized_unscaled = sdf_field_to_image(live_field, scale=1)
    cv2.imwrite(os.path.join(out_path, "unscaled_initial_live.png"), live_visualized_unscaled)
    cv2.imwrite(os.path.join(out_path, "initial_live.png"), live_visualized)


def visualize_final_fields(canonical_field, live_field, view_scaling_factor):
    cv2.imshow("live SDF", sdf_field_to_image(live_field, scale=view_scaling_factor))
    process_cv_esc()
    cv2.imshow("canonical SDF", sdf_field_to_image(canonical_field, scale=view_scaling_factor))
    process_cv_esc()
    cv2.destroyAllWindows()


def save_final_fields(canonical_field, live_field, out_path, view_scaling_factor):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    final_live = sdf_field_to_image(live_field, scale=view_scaling_factor)
    cv2.imwrite(os.path.join(out_path, 'final_live.png'), final_live)
    final_canonical = sdf_field_to_image(canonical_field, scale=view_scaling_factor)
    cv2.imwrite(os.path.join(out_path, "final_canonical.png"), final_canonical)
