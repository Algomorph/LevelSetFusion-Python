#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  For transformation in 2D.
#  ================================================================

import math
import numpy as np
import cv2


def twist_vector_to_matrix2d(twist):
    # for transforming translation and rotation vector to homo matrix in 2D

    theta = twist[2]
    twist_matrix = np.identity(3)
    twist_matrix[0, 0] = math.cos(theta)
    twist_matrix[0, 1] = -math.sin(theta)
    twist_matrix[1, 0] = math.sin(theta)
    twist_matrix[1, 1] = math.cos(theta)
    twist_matrix[0, 2] = twist[0]
    twist_matrix[1, 2] = twist[1]

    return twist_matrix  # 3 by 3 matrix


def twist_vector_to_matrix3d(twist):
    # for transforming translation and rotation vector to homo matrix in 3D

    twist_matrix = cv2.Rodrigues(twist[3:6])[0]
    twist_matrix = np.concatenate((twist_matrix, np.zeros((1, 3))), axis=0)
    twist_matrix = np.concatenate((twist_matrix, np.array([twist[0], twist[1], twist[2], [1]])), axis=1)

    return twist_matrix  # 4 by 4 matrix