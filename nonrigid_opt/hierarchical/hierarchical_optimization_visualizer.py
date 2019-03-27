#  ================================================================
#  Created by Gregory Kramida on 12/3/18.
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

# contains main visualization subroutines for showing and recording results of the optimization process in the
# HierarchicalNonrigidSLAMOptimizer2d class

# stdlib
import os.path
import os
# libraries
import cv2
import numpy as np
# local
from attr import has

import utils.visualization as viz
import level_set_fusion_optimization as ho_cpp


class HierarchicalOptimizer2dVisualizer:
    class Parameters:
        def __init__(self, out_path="output/ho", view_scaling_fator=8,
                     show_live_progression=False,
                     save_live_progression=False,
                     save_initial_fields=False,
                     save_final_fields=False,
                     save_warp_field_progression=False,
                     save_data_gradients=False,
                     save_tikhonov_gradients=False):
            self.out_path = out_path
            self.view_scaling_factor = view_scaling_fator
            self.show_live_progress = show_live_progression

            self.save_live_field_progression = save_live_progression
            self.save_initial_fields = save_initial_fields
            self.save_final_fields = save_final_fields
            self.save_warp_field_progression = save_warp_field_progression
            self.save_data_gradients = save_data_gradients
            self.save_tikhonov_gradients = save_tikhonov_gradients
            self.using_output_folder = self.save_final_fields or \
                                       self.save_initial_fields or \
                                       self.save_live_field_progression or \
                                       self.save_warp_field_progression or \
                                       self.save_data_gradients or \
                                       self.save_tikhonov_gradients

    def __init__(self, parameters=None, field_size=128, level_count=4):
        self.field_size = field_size
        self.parameters = parameters
        self.level_count = level_count
        if not parameters:
            self.parameters = HierarchicalOptimizer2dVisualizer.Parameters()
        # initialize video-writers
        self.live_progression_writer = None
        self.warp_video_writer2D = None
        self.data_gradient_video_writer2D = None
        self.tikhonov_gradient_video_writer2D = None

        if self.parameters.using_output_folder:
            if not os.path.exists(self.parameters.out_path):
                os.makedirs(self.parameters.out_path)

        if self.parameters.save_live_field_progression:
            self.live_progression_writer = cv2.VideoWriter(
                os.path.join(self.parameters.out_path, 'live_field_evolution_2D.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
                (field_size * self.parameters.view_scaling_factor, field_size * self.parameters.view_scaling_factor),
                isColor=False)
        if self.parameters.save_warp_field_progression:
            self.warp_video_writer2D = cv2.VideoWriter(
                os.path.join(self.parameters.out_path, 'warp_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)

        if self.parameters.save_data_gradients:
            self.data_gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.parameters.out_path, 'data_gradient_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)

        if self.parameters.save_tikhonov_gradients:
            self.tikhonov_gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.parameters.out_path, 'tikhonov_gradient_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)

    def generate_pre_optimization_visualizations(self, canonical_field, live_field):
        if self.parameters.save_initial_fields:
            viz.save_initial_fields(canonical_field, live_field, self.parameters.out_path,
                                    self.parameters.view_scaling_factor)

    def generate_post_optimization_visualizations(self, canonical_field, live_field, warp_field):
        if self.parameters.save_final_fields:
            viz.save_final_fields(canonical_field, live_field, self.parameters.out_path,
                                  self.parameters.view_scaling_factor)

    def generate_per_iteration_visualizations(self, hierarchy_level_index, iteration_number,
                                              canonical_field, live_field, warp_field,
                                              data_gradient=None, inverse_tikhonov_gradient=None):
        level_scaling = 2 ** (self.level_count - hierarchy_level_index - 1)
        if self.parameters.save_live_field_progression:
            live_field_out = viz.sdf_field_to_image(live_field, self.parameters.view_scaling_factor * level_scaling)
            self.live_progression_writer.write(live_field_out)

        if self.parameters.save_warp_field_progression:
            upscaled_warp_field = warp_field.repeat(level_scaling, axis=0).repeat(level_scaling, axis=1)
            self.warp_video_writer2D.write(
                viz.make_vector_field_plot(upscaled_warp_field, scale=1.0, iteration_number=iteration_number,
                                           vectors_name="Warp vectors"))
        if self.parameters.save_data_gradients:
            upscaled_data_gradient = data_gradient.repeat(level_scaling, axis=0).repeat(level_scaling, axis=1)
            self.data_gradient_video_writer2D.write(
                viz.make_vector_field_plot(upscaled_data_gradient, scale=10.0, iteration_number=iteration_number,
                                           vectors_name="Data gradient (10X magnitude)"))
        if self.parameters.save_tikhonov_gradients:
            if inverse_tikhonov_gradient is None:
                raise ValueError("Expected a numpy array for inverse_tikhonov_gradient, got None. "
                                 "Is Tikhonov term enabled for the calling optimizer?")
            upscaled_tikhonov_gradient = \
                inverse_tikhonov_gradient.repeat(level_scaling, axis=0).repeat(level_scaling, axis=1)
            self.tikhonov_gradient_video_writer2D.write(
                viz.make_vector_field_plot(upscaled_tikhonov_gradient, scale=10.0, iteration_number=iteration_number,
                                           vectors_name="Inverse tikhonov gradient (10X magnitude)"))

    def __del__(self):
        if self.live_progression_writer:
            self.live_progression_writer.release()
        if self.warp_video_writer2D:
            self.warp_video_writer2D.release()
        if self.data_gradient_video_writer2D:
            self.data_gradient_video_writer2D.release()
        if self.tikhonov_gradient_video_writer2D:
            self.tikhonov_gradient_video_writer2D.release()


class TelemetryMetadata:
    def __init__(self, has_warp_fields, has_data_term_gradients, has_tikhonov_term_gradients, field_size):
        self.has_warp_fields = has_warp_fields
        self.has_data_term_gradients = has_data_term_gradients
        self.has_tikhonov_term_gradients = has_tikhonov_term_gradients
        self.field_size = field_size


class OptimizationIterationData:
    def __init__(self, warp_fields, data_term_gradients, tikhonov_term_gradients):
        self.warp_fields = warp_fields
        self.data_term_gradients = data_term_gradients
        self.tikhonov_term_gradients = tikhonov_term_gradients

    def get_warp_fields(self):
        return self.warp_fields

    def get_data_term_gradients(self):
        return self.data_term_gradients

    def get_tikhonov_term_gradients(self):
        return self.tikhonov_term_gradients

    def get_frame_count(self):
        return len(self.warp_fields)


def get_number_of_frames_to_save_from_telemetry_logs(telemetry_logs):
    frame_count = 0
    for log in telemetry_logs:
        for level_data in log:
            frame_count += level_data.get_frame_count()
    return frame_count


def get_telemetry_metadata(telemetry_log):
    first_level_data = telemetry_log[0]
    field_size = 0
    has_warp_fields = False
    has_data_term_gradients = False
    has_tikhonov_term_gradients = False
    warp_fields = first_level_data.get_warp_fields()
    if warp_fields is not None and len(warp_fields) > 0:
        first_warp_field = warp_fields[0]
        if first_warp_field is not None and first_warp_field.size > 0:
            has_warp_fields = True
            field_size = first_warp_field.shape[0]
    data_term_gradients = first_level_data.get_data_term_gradients()
    if data_term_gradients is not None and len(data_term_gradients) > 0:
        first_data_term_gradient = data_term_gradients[0]
        if first_data_term_gradient is not None and first_data_term_gradient.size > 0:
            has_data_term_gradients = True
            if field_size == 0:
                field_size = first_data_term_gradient.shape[0]
    tikhonov_term_gradients = first_level_data.get_tikhonov_term_gradients()
    if tikhonov_term_gradients is not None and len(tikhonov_term_gradients):
        first_tikhonov_term_gradient = tikhonov_term_gradients[0]
        if first_tikhonov_term_gradient is not None and first_tikhonov_term_gradient.size > 0:
            has_data_term_gradients = True
            if field_size == 0:
                field_size = first_tikhonov_term_gradient.shape[0]
    return TelemetryMetadata(has_warp_fields, has_data_term_gradients, has_tikhonov_term_gradients, field_size)


def save_telemetry_log(telemetry_log, telemetry_metadata, output_folder):
    """
    :param telemetry_log: telemetry log to save
    :type telemetry_metadata: TelemetryMetadata
    :param telemetry_metadata:
    :param output_folder:
    :return:
    """
    telemetry_dict = {}
    for i_level, level_data in enumerate(telemetry_log):
        telemetry_dict["l{:d}_warp_fields".format(i_level)] = np.dstack(level_data.get_warp_fields())
        telemetry_dict["l{:d}_data_term_gradients".format(i_level)] = \
            np.array([]) if not telemetry_metadata.has_data_term_gradients else np.dstack(
                level_data.get_data_term_gradients())
        telemetry_dict["l{:d}_tikhonov_term_gradients".format(i_level)] = \
            np.array([]) if not telemetry_metadata.has_tikhonov_term_gradients else np.dstack(
                level_data.get_tikhonov_term_gradients())
    np.savez_compressed(os.path.join(output_folder, "telemetry_log.npz"), **telemetry_dict)


def load_telemetry_log(output_folder):
    path = os.path.join(output_folder, "telemetry_log.npz")
    telemetry_dict = np.load(path)
    level_count = len(telemetry_dict.files) // 3
    telemetry_log = []
    for i_level in range(level_count):
        warp_fields = telemetry_dict["l{:d}_warp_fields".format(i_level)]
        warp_fields = np.dsplit(warp_fields, warp_fields.shape[2]//2)
        data_term_gradients = telemetry_dict["l{:d}_data_term_gradients".format(i_level)]
        if len(data_term_gradients.shape) == 3:
            data_term_gradients = np.dsplit(data_term_gradients, data_term_gradients.shape[2]//2)
        else:
            data_term_gradients = (np.array([])) * len(warp_fields)
        tikhonov_term_gradients = telemetry_dict["l{:d}_tikhonov_term_gradients".format(i_level)]
        if len(tikhonov_term_gradients.shape) == 3:
            tikhonov_term_gradients = np.dsplit(tikhonov_term_gradients, tikhonov_term_gradients.shape[2]//2)
        else:
            tikhonov_term_gradients = (np.array([])) * len(warp_fields)
        level_data = OptimizationIterationData(warp_fields, data_term_gradients, tikhonov_term_gradients)
        telemetry_log.append(level_data)
    return telemetry_log


def convert_cpp_telemetry_logs_to_video(telemetry_log, telemetry_metadata, canonical_field, live_field, output_folder,
                                        progress_bar=None):
    """
    :type telemetry_metadata: TelemetryMetadata
    :param telemetry_metadata: metadata that tells what kinds of data the telemetry log contains
    :param canonical_field: canonical field (typically, warp target)
    :param live_field: initial live field (typically, warp source)
    :param output_folder:
    :type telemetry_log: ho_cpp.OptimizationIterationDataVector
    :param telemetry_log: intermediate results for each level for each iteration
    :type progress_bar: progressbar.ProgressBar
    :param progress_bar: optional progress bar
    :return:
    """
    has_warp_fields = telemetry_metadata.has_warp_fields
    has_data_term_gradients = telemetry_metadata.has_data_term_gradients
    has_tikhonov_gradients = telemetry_metadata.has_tikhonov_term_gradients

    if not has_warp_fields:
        return

    field_size = telemetry_metadata.field_size

    scaling_factor = 1024 / field_size

    level_count = len(telemetry_log)

    params = HierarchicalOptimizer2dVisualizer.Parameters(
        out_path=output_folder,
        view_scaling_fator=scaling_factor,
        show_live_progression=False,
        save_live_progression=False,
        save_initial_fields=False,
        save_final_fields=False,
        save_warp_field_progression=has_warp_fields,
        save_data_gradients=has_data_term_gradients,
        save_tikhonov_gradients=has_tikhonov_gradients
    )
    visualizer = HierarchicalOptimizer2dVisualizer(params, field_size, level_count)

    i_frame = 0
    if progress_bar is not None:
        i_frame = progress_bar.value
    for i_level, level_data in enumerate(telemetry_log):
        frame_count = level_data.get_frame_count()
        warp_fields = level_data.get_warp_fields()
        data_term_gradients = level_data.get_data_term_gradients()
        tikhonov_gradients = level_data.get_tikhonov_term_gradients()
        for i_iteration in range(frame_count):
            warp_field = warp_fields[i_iteration]
            data_term_gradient = None if not has_data_term_gradients else data_term_gradients[i_iteration]
            tikhonov_gradient = None if not has_tikhonov_gradients else tikhonov_gradients[i_iteration]
            visualizer.generate_per_iteration_visualizations(i_level, i_iteration, canonical_field, live_field,
                                                             warp_field, data_term_gradient, tikhonov_gradient)
            if progress_bar is not None:
                progress_bar.update(i_frame)
            i_frame += 1
