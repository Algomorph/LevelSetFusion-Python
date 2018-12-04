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


class HNSOVisualizer:
    class HNSOVisualizerSettings:
        def __init__(self, show_live_progress=False, save_live_progress=False):
            self.show_live_progress = show_live_progress
            self.save_live_progress = save_live_progress

    def __init__(self):
        # TODO
        pass
