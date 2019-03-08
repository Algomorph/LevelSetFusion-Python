#  ================================================================
#  Created by Gregory Kramida on 3/8/19.
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

# This is just a convenience routine that utilizes a hardcoded dictionary of locations that determine location of the
# "Reconstruction" folder on different developer/maintainer team machines

import socket

paths_by_machine_name = {"june-ubuntu": "/mnt/4696C5EE7E51F6BB/Reconstruction",
                         "Juggernaut": "/media/algomorph/Data/Reconstruction"}


def get_reconstruction_directory():
    hostname = socket.gethostname()

    if hostname in paths_by_machine_name:
        return paths_by_machine_name[hostname]

    return "/media/algomorph/Data/Reconstruction"
