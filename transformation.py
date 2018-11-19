#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  For transformation in 2D.
#  ================================================================
import math
import numpy as np


def twist_vector_to_matrix(twist):
    # for transforming translation and rotation vector to homo matrix in 2D

    # rotation3DMatrix = cv2.Rodrigues(rotation3DVector)

    theta = twist[2]
    twist_matrix_homo = np.identity(3)
    twist_matrix_homo[0, 0] = math.cos(theta)
    twist_matrix_homo[0, 1] = -math.sin(theta)
    twist_matrix_homo[1, 0] = math.sin(theta)
    twist_matrix_homo[1, 1] = math.cos(theta)
    twist_matrix_homo[0, 2] = twist[0]
    twist_matrix_homo[1, 2] = twist[1]

    return twist_matrix_homo  # 3 by 3 matrix
