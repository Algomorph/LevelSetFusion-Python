#  ================================================================
#  Created by Gregory Kramida on 12/4/18.
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
# stdlib
import os
# libraries
import cv2
import numpy as np

from utils.visualization import make_3d_plots, sdf_field_to_image, make_vector_field_plot, warp_field_to_heatmap


class SlavchevaVisualizer:
    class Settings:
        def __init__(self, view_scaling_factor=8,
                     enable_component_fields=False,
                     enable_3d_plot=False,
                     enable_warp_quiverplot=True,
                     enable_gradient_quiverplot=True,
                     level_set_term_enabled=False
                     ):
            # visualization flags & parameters
            self.enable_3d_plot = enable_3d_plot
            self.enable_warp_quiverplot = enable_warp_quiverplot
            self.enable_gradient_quiverplot = enable_gradient_quiverplot
            self.enable_component_fields = enable_component_fields
            self.view_scaling_factor = view_scaling_factor
            self.level_set_term_enabled = level_set_term_enabled

    def __init__(self, field_size, out_path, settings=None):
        if settings:
            self.settings = settings
        else:
            self.settings = SlavchevaVisualizer.Settings()

        self.out_path = out_path

        self.data_component_field = None
        self.smoothing_component_field = None
        self.level_set_component_field = None

        # prepare to log fields for warp vector components from various terms if necessary
        if self.settings.enable_component_fields:
            self.data_component_field = np.zeros((field_size, field_size, 2), dtype=np.float32)
            self.smoothing_component_field = np.zeros((field_size, field_size, 2), dtype=np.float32)
            if self.settings.level_set_term_enabled:
                self.level_set_component_field = np.zeros((field_size, field_size, 2), dtype=np.float32)
            else:
                self.level_set_component_field = None

        self.live_video_writer2D = None
        self.warp_magnitude_video_writer2D = None
        self.live_video_writer3D = None
        self.warp_video_writer2D = None
        self.gradient_video_writer2D = None
        self.data_gradient_video_writer2D = None
        self.smoothing_gradient_video_writer2D = None
        self.level_set_gradient_video_writer2D = None

        # video writers
        self.live_video_writer2D = cv2.VideoWriter(
            os.path.join(self.out_path, 'live_field_evolution_2D.mkv'),
            cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
            (field_size * self.settings.view_scaling_factor, field_size * self.settings.view_scaling_factor),
            isColor=False)
        self.warp_magnitude_video_writer2D = cv2.VideoWriter(
            os.path.join(self.out_path, 'warp_magnitudes_2D.mkv'),
            cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
            (field_size * self.settings.view_scaling_factor, field_size * self.settings.view_scaling_factor),
            isColor=True)
        if self.settings.enable_3d_plot:
            self.live_video_writer3D = cv2.VideoWriter(
                os.path.join(self.out_path, 'live_field_evolution_2D_3D_plot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1230, 720), isColor=True)
        if self.settings.enable_warp_quiverplot:
            self.warp_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'warp_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
        if self.settings.enable_gradient_quiverplot:
            self.gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'gradient_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
        if self.settings.enable_component_fields:
            self.data_gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'data_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
            self.smoothing_gradient_video_writer2D = cv2.VideoWriter(
                os.path.join(self.out_path, 'smoothing_2D_quiverplot.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)
            if self.settings.level_set_term_enabled:
                self.level_set_gradient_video_writer2D = cv2.VideoWriter(
                    os.path.join(self.out_path, 'level_set_2D_quiverplot.mkv'),
                    cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10, (1920, 1200), isColor=True)

    def __del__(self):
        if self.live_video_writer3D is not None:
            self.live_video_writer3D.release()
        if self.warp_video_writer2D is not None:
            self.warp_video_writer2D.release()
        if self.gradient_video_writer2D is not None:
            self.gradient_video_writer2D.release()
        if self.data_gradient_video_writer2D is not None:
            self.data_gradient_video_writer2D.release()
        if self.smoothing_gradient_video_writer2D is not None:
            self.smoothing_gradient_video_writer2D.release()
        if self.level_set_gradient_video_writer2D is not None:
            self.level_set_gradient_video_writer2D.release()

        self.live_video_writer2D.release()
        self.warp_magnitude_video_writer2D.release()

    def write_live_sdf_visualizations(self, canonical_field, live_field):
        if self.live_video_writer3D is not None:
            make_3d_plots(self.live_video_writer3D, canonical_field, live_field)
        if self.live_video_writer2D is not None:
            self.live_video_writer2D.write(sdf_field_to_image(live_field, self.settings.view_scaling_factor))
        pass

    def write_all_iteration_visualizations(self, iteration_number, warp_field, gradient_field, live_field,
                                           canonical_field):
        if self.warp_video_writer2D is not None:
            make_vector_field_plot(self.warp_video_writer2D, warp_field,
                                   scale=10.0, iteration_number=iteration_number,
                                   vectors_name="Warp vectors (scaled x10)")
        if self.gradient_video_writer2D is not None:
            make_vector_field_plot(self.gradient_video_writer2D, -gradient_field, iteration_number=iteration_number,
                                   vectors_name="Gradient vectors (negated)")
        if self.data_gradient_video_writer2D is not None:
            make_vector_field_plot(self.data_gradient_video_writer2D, -self.data_component_field,
                                   iteration_number=iteration_number, vectors_name="Data gradients (negated)")
        if self.smoothing_gradient_video_writer2D is not None:
            make_vector_field_plot(self.smoothing_gradient_video_writer2D, -self.smoothing_component_field,
                                   iteration_number=iteration_number, vectors_name="Smoothing gradients (negated)")
        if self.level_set_gradient_video_writer2D is not None:
            make_vector_field_plot(self.level_set_gradient_video_writer2D, -self.level_set_component_field,
                                   iteration_number=iteration_number, vectors_name="Level set gradients (negated)")

        if self.live_video_writer2D is not None:
            self.live_video_writer2D.write(sdf_field_to_image(live_field, self.settings.view_scaling_factor))
        if self.warp_magnitude_video_writer2D is not None:
            self.warp_magnitude_video_writer2D.write(
                warp_field_to_heatmap(warp_field, self.settings.view_scaling_factor))
        if self.live_video_writer3D is not None:
            make_3d_plots(self.live_video_writer3D, canonical_field, live_field)
