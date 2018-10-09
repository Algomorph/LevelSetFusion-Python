#!/usr/bin/python3
#  ================================================================
#  Created by Gregory Kramida on 2/1/18.
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

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

EXIT_STATUS_SUCCESS = 0
EXIT_STATUS_FAILURE = 1

def main(args):
    df = pd.read_csv("/media/algomorph/Data/Reconstruction/debug_output/energy_stats.txt")
    sample_data_table = FF.create_table(df.head())
    py.iplot(sample_data_table, filename="/media/algomorph/Data/Reconstruction/debug_output/energy_stats.png")
    return EXIT_STATUS_SUCCESS


if __name__ == "__main__":
    sys.exit(main(sys.argv))
