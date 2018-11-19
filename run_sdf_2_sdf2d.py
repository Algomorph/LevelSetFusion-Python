#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  Rigid alignment algorithm implementation based on SDF-2-SDF paper.
#  ================================================================


import tsdf_field_generation as tfg
import numpy as np
from dataset import datasets, DataToUse
from utils.vizualization import visualize_and_save_initial_fields, visualize_final_fields


def main():
    default_value = 1
    data_to_use = DataToUse.SIMPLE_TEST_CASE01
    live_field, canonical_field = datasets[data_to_use].generate_2d_sdf_fields(default_value)
    field_size = datasets[data_to_use].field_size



if __name__ == "__main__":
    main()
