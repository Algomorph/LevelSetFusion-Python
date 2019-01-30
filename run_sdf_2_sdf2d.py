#  ================================================================
#  Created by Fei Shan on 11/07/18.
#  Rigid alignment algorithm implementation based on SDF-2-SDF paper.
#  ================================================================

# stdlib
import sys

import numpy as np
import cv2
import matplotlib.pyplot as plt

# local
from rigid_opt import sdf_2_sdf_visualizer as sdf2sdfv, sdf_2_sdf_optimizer2d as sdf2sdfo
from rigid_opt.transformation import twist_vector_to_matrix
from experiment import dataset as ds
from tsdf import generation as tsdf
from utils import field_resampling as resampling
import utils.sampling as sampling
from tsdf import generation as tsdf_gen
from calib.camera import Camera, DepthCamera
from calib.geom import Pose

EXIT_CODE_SUCCESS = 0
EXIT_CODE_FAILURE = 1


class ImageBasedSingleFrameDataset:
    def __init__(self, first_frame_path, second_frame_path, image_pixel_row, field_size, offset, camera):
        self.first_frame_path = first_frame_path
        self.second_frame_path = second_frame_path
        self.image_pixel_row = image_pixel_row
        self.field_size = field_size
        self.offset = offset
        self.depth_camera = camera

    def generate_2d_sdf_fields(self, method=tsdf_gen.DepthInterpolationMethod.NONE):

        depth_image0 = cv2.imread(self.first_frame_path, -1)
        depth_image0 = cv2.cvtColor(depth_image0, cv2.COLOR_BGR2GRAY)
        depth_image0 = depth_image0.astype(float) # convert to meters

        # max_depth = np.iinfo(np.uint16).max
        # depth_image0[depth_image0 == 0] = max_depth
        canonical_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image0, self.depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size,
                                                             # default_value=-999.,
                                                             array_offset=self.offset,
                                                             narrow_band_width_voxels=1.,
                                                             depth_interpolation_method=method)

        depth_image1 = cv2.imread(self.first_frame_path, -1)
        depth_image1 = cv2.cvtColor(depth_image1, cv2.COLOR_BGR2GRAY)
        depth_image1 = depth_image1.astype(float)  # convert to meters

        # depth_image1[depth_image1 == 0] = max_depth
        live_field = \
            tsdf_gen.generate_2d_tsdf_field_from_depth_image(depth_image1, self.depth_camera, self.image_pixel_row,
                                                             field_size=self.field_size,
                                                             # default_value=-999.,
                                                             array_offset=self.offset,
                                                             narrow_band_width_voxels=1.,
                                                             depth_interpolation_method=method)
        return live_field, canonical_field


def main():
    canonical_frame_path = "../Data/Synthetic_Kenny_Circle/depth_000000.exr"
    live_frame_path = "../Data/Synthetic_Kenny_Circle/depth_000003.exr"
    image_pixel_row = 240

    intrinsic_matrix = np.matrix([[570.3999633789062, 0, 320],  # FX = 570.3999633789062 CX = 320.0
                                  [0, 570.3999633789062, 240],  # FY = 570.3999633789062 CY = 240.0
                                  [0, 0, 1]], dtype=np.float32)
    camera = DepthCamera(intrinsics=DepthCamera.Intrinsics(resolution=(480, 640),
                                                           intrinsic_matrix=intrinsic_matrix))
    offset = np.array([-16, -16, 104])
    data_to_use = ImageBasedSingleFrameDataset(
        canonical_frame_path,  # dataset from original sdf2sdf paper, reference frame
        live_frame_path,  # dataset from original sdf2sdf paper, current frame
        image_pixel_row, 32, offset, camera
    )

    live_depth_image = cv2.imread(live_frame_path, -1)
    live_depth_image = cv2.cvtColor(live_depth_image, cv2.COLOR_BGR2GRAY)
    live_depth_image1d = live_depth_image.astype(float)[image_pixel_row] # convert to meters

    depth_interpolation_method = tsdf.DepthInterpolationMethod.NONE
    out_path = "output/sdf2sdf"
    sampling.set_focus_coordinates(0, 0)
    generate_test_data = False
    live_field, canonical_field = data_to_use.generate_2d_sdf_fields(method=depth_interpolation_method)
    print(canonical_field.max(), canonical_field.min())

    optimizer = sdf2sdfo.Sdf2SdfOptimizer2d(
        verbosity_parameters=sdf2sdfo.Sdf2SdfOptimizer2d.VerbosityParameters(print_max_warp_update=True,
                                                                             print_iteration_energy=True),
        visualization_parameters=sdf2sdfv.Sdf2SdfVisualizer.Parameters(out_path=out_path)
        )
    optimizer.optimize(live_depth_image1d, canonical_field, live_field, camera, offset, iteration=10)

    return EXIT_CODE_SUCCESS









    DIMENSION = 2 # 2D case
    filename0 = "../Data/Synthetic_Kenny_Circle/depth_000000.exr" # dataset from original sdf2sdf paper, referance
    filename1 = "../Data/Synthetic_Kenny_Circle/depth_000003.exr" # dataset from original sdf2sdf paper, current

    FX = 570.3999633789062
    FY = 570.3999633789062
    CX = 320.0
    CY = 240.0
    SCALE = 0.001 # from depth value to meter
    ROWSLICE = int(CY)

    DELTA = .002  # truncated SDF threshold
    ETA = .01  # thickness of surface
    L = .004  # side length
    BETA = 0.5  # step size of twist

    # load image and find the corresponding points in space
    depthImage0 = load_exr_file(filename0, SCALE, DIMENSION, ROWSLICE)  # load the depth image, (640, )
    # print depthImage0
    depthImage1 = load_exr_file(filename1, SCALE, DIMENSION, ROWSLICE)
    reverseProjection0 = calculateReverseProjection(depthImage0, FX, CX)  # get location of point in space for each pixel, (x, z), (640, 2)
    reverseProjection1 = calculateReverseProjection(depthImage1, FX, CX)
    # print reverseProjection0

    # get the bound of the volume
    lowerLeft = (np.min(reverseProjection0[np.nonzero(reverseProjection0[:, 0]), 0]) - 2 * L, np.min(reverseProjection0[np.nonzero(reverseProjection0[:, 1]), 1]) - 2 * L) # get the lower bound of volume except for zero
    upperRight = (np.amax(reverseProjection0[:, 0] + 2 * L), np.amax(reverseProjection0[:, 1] + 2 * L)) # get the upper bound of volume

    C = (upperRight[1], lowerLeft[0]) # consider as origin of the voxel volume

    # optimization initialization for each point
    maxVoxelIndex = calculateVoxelIndex(upperRight, lowerLeft, L) # vox in the paper
    mMax, nMax = int(maxVoxelIndex[1]), int(maxVoxelIndex[0]) # voxel size
    # print mMax, nMax     # 12 65, (x, z)

    twist = np.zeros((DIMENSION+1, 1)) # two for translation and one for rotation, (3, 1)
    refWeight, curWeight = 0, 0

    # for camera pose estimation
    # errorArray = np.zeros(60)
    for iter in range(60):
        matrixA = np.zeros((DIMENSION+1, DIMENSION+1)) # A in the paper
        matrixb = np.zeros((DIMENSION+1, 1)) # b in the paper
        error = 0. # EGeometry in the paper
        # loop through voxels, for each voxel we consider its center point
        for i in range(mMax):
            for j in range(nMax): # loop through each voxel
                voxel = calculateVoxel((i, j), C, L) # V in the paper
                refSDF, refWeight = SDFAndWeight(depthImage0, voxel, twist, DELTA, ETA, FX, CX, False)
                curSDF, curWeight = SDFAndWeight(depthImage1, voxel, twist, DELTA, ETA, FX, CX, True)
                # print refSDF, curSDF
                if refSDF < -1 or curSDF < -1:
                    continue
                curGradient = SDFGradient(depthImage1, voxel, twist, DELTA, ETA, FX, CX)
                # print curGradient
                matrixA += np.dot(curGradient.T, curGradient)
                matrixb += (refSDF - curSDF + np.dot(curGradient, twist)) * curGradient.T
                error += 0.5 * (refSDF * refWeight - curSDF * curWeight)**2
                # print errors
        # print "A : \n", matrixA, "\n b : \n", matrixb
        # errorArray[iter] = error
        twistStar = np.dot(np.linalg.inv(matrixA), matrixb)
        twist += BETA * (twistStar - twist)
        # print "optimal twist is ", twistStar
        print("error(/Egeom): ", error, "at iteration ", iter)
        print("twist vector is \n", twist)

        # if iter > 0 and abs(errorArray[iter] - errorArray[iter-1]) <= 0.00000001:
            # break
    # print errorArray
    plotPoints(reverseProjection0, reverseProjection1, twist)


def load_exr_file(filename, scale, dimension, row): # load file and convert to world corrdinate size
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    image = image.astype(float)*scale # conver pixel value to meters
    return image[row] # for 2D cases, image is a line of pixels


def calculateReverseProjection(depthImage, fx, cx): # project pixel value to points in the volume
    projMap = np.zeros((depthImage.size, 2)) # store corresponding vertex location in eath pixel
    for i in range(depthImage.size):
        projMap[i, 0] = (i - cx) * depthImage[i] / fx
        projMap[i, 1] = depthImage[i]
    return projMap


def calculateVoxelIndex(p, c, l): # find the corresponding voxel index for arbitrary point
    voxelIndex = np.array([float(round((1./l)*(p[0]-c[0])-.5)), float(round((1./l)*(p[1]-c[1])-.5))])
    return voxelIndex


def calculateVoxel(v, c, l): # consider points in the voxel as the voxel center point
    voxel = np.array([float(l*(v[0]+.5)+c[0]), float(l*(v[1]+.5)+c[1])])
    return voxel


def calculateTrueSDF(depthImage, i, voxelCenter): # true SDF for a point
    return depthImage[i] - voxelCenter[1]


def calculateSDF(trueSDF, delta): # truncated SDF for a point
    if trueSDF >= delta:
        return 1
    elif trueSDF <= -1. * delta:
        return -1
    else:
        return trueSDF/delta


def calcualteWeight(trueSDF, eta): # weight of a SDF
    if trueSDF > -1. * eta:
        return 1
    else:
        return 0


def SDFAndWeight(depthImage, voxel, twist, delta, eta, fx, cx, needTwist): # Signed Distance and Weight for each point
    # transform to image coordinates
    # print voxel
    homogeneousTrans = twist_vector_to_matrix(twist)
    # print homogeneousTrans
    voxelHomo = np.array([voxel[0], voxel[1], 1]) # voxel center in homogeneous
    if needTwist:
        voxelHomo = np.dot(homogeneousTrans, voxelHomo) #tranformation for current frame voxel houmogenous
    voxelImageCoor = int(fx*voxelHomo[0]/voxelHomo[1] + cx) # projection and rasterization to get the pixel coordinates
    # print voxelImageCoor
    if voxelImageCoor < 0 or voxelImageCoor >= depthImage.size:
        return -999, 0
    trueSDF = calculateTrueSDF(depthImage, voxelImageCoor, voxel)
    # print "trueSDF: ", trueSDF
    SDF = calculateSDF(trueSDF, delta)
    # print "SDF: ", SDF
    weight = calcualteWeight(trueSDF, eta)
    # print "wieght: ", weight
    return SDF, weight

def SDFGradient(depthImage, voxel, twist, delta, eta, fx, cx): # gradient respect to twist implemented as partial derivative
    epsilon = 4.0e-3
    DIMENSION = 2
    gradient = np.zeros((1, DIMENSION))
    # preTwist = np.zeros((DIMENSION+1, 1))
    # postTwist = np.zeros((DIMENSION+1, 1))

    preX = voxel
    preX[0] = voxel[0] - epsilon # at X direction
    prePhi, preWeight = SDFAndWeight(depthImage, preX, twist, delta, eta, fx, cx, False)
    postX = voxel
    postX[0] = voxel[0] + epsilon # at X direction
    postPhi, postWeight = SDFAndWeight(depthImage, postX, twist, delta, eta, fx, cx, False)
    if (postPhi < -1 ) or (prePhi < -1):
        gradient[0, 0] = 0
    else:
        gradient[0, 0] = (postPhi - prePhi)/ (2. * epsilon)

    preZ = voxel
    preZ[1] = voxel[1] - epsilon # at Z direction
    prePhi, preWeight = SDFAndWeight(depthImage, preZ, twist, delta, eta, fx, cx, False)
    postZ = voxel
    postZ[1] = voxel[1] + epsilon # at Z direction
    postPhi, postWeight = SDFAndWeight(depthImage, postZ, twist, delta, eta, fx, cx, False)
    if postPhi < -1 or prePhi < -1:
        gradient[0, 1] = 0
    else:
        gradient[0, 1] = (postPhi - prePhi)/(2. * epsilon)

    twistMatrixHomo = twist_vector_to_matrix(twist)
    voxelTransformedHomo = np.dot(np.linalg.inv(twistMatrixHomo), np.concatenate((voxel.T, np.ones(1))))
    voxelTransformed = np.delete(voxelTransformedHomo, 2)
    concatenetMatrix = np.concatenate((np.identity(2), np.array([[-voxelTransformed[1]], [voxelTransformed[0]]])), axis = 1)
    # print concatenetMatrix, gradient, np.dot(gradient, concatenetMatrix)
    return np.dot(gradient, concatenetMatrix) # (1, 3)

def plotPoints(reverseProjection0, reverseProjection1, twist): # plot points of each pixel to the volume
    reverseProjection1 = np.concatenate((reverseProjection1, np.ones((reverseProjection1.shape[0], 1))), axis = 1)
    for i in range(reverseProjection1.shape[0]):
        reverseProjection1[i, 0] = np.dot(twist_vector_to_matrix(twist), reverseProjection1[i, :].T)[0]
        reverseProjection1[i, 1] = np.dot(twist_vector_to_matrix(twist), reverseProjection1[i, :].T)[1]
    np.delete(reverseProjection1, 2, axis = 0)
    plt.plot(reverseProjection0[:, 0], reverseProjection0[:, 1], 'b.', reverseProjection1[:, 0], reverseProjection1[:, 1], 'r.')
    plt.show()
if __name__ == "__main__":
    main()
