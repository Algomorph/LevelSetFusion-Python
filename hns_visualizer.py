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
# libraries
import cv2


class HNSOVisualizer:
    class HNSOVisualizerSettings:
        def __init__(self, out_path="output/hns_optimizer/", view_scaling_fator=8,
                     show_live_progress=False, save_live_progress=False):
            self.out_path = out_path
            self.view_scaling_factor = view_scaling_fator
            self.show_live_progress = show_live_progress
            self.save_live_progress = save_live_progress

    def __init__(self, settings=None, field_size=128):
        self.field_size = field_size
        self.settings = settings
        if not settings:
            self.settings = HNSOVisualizer.HNSOVisualizerSettings()
        # initialize video-writers
        self.live_progression_writer = None
        if settings.save_live_progress:
            self.live_progression_writer = cv2.VideoWriter(
                os.path.join(self.settings.out_path, 'live_field_evolution_2D.mkv'),
                cv2.VideoWriter_fourcc('X', '2', '6', '4'), 10,
                (field_size * self.settings.view_scaling_factor, field_size * self.settings.view_scaling_factor),
                isColor=False)
