import numpy as np
from math import *

def toSolveFullMatrix(point_des, point_src, npoints): # right, left

    ## solve the matrix from des to src (right to left)
    x_des = point_des[0, :].T  # col
    y_des = point_des[1, :].T  # row
    x_src = point_src[0, :].T
    y_src = point_src[1, :].T

    A = np.zeros((npoints * 2, 8))  # 2N * 8
    B = np.ones((npoints * 2, 1))

    for i in range(npoints):
        A[2 * i][0] = x_des[i]
        A[2 * i][1] = y_des[i]
        A[2 * i][2] = 1
        A[2 * i][6] = -x_des[i] * x_src[i]
        A[2 * i][7] = -y_des[i] * x_src[i]
        B[2 * i][0] = x_src[i]

        A[2 * i + 1][3] = x_des[i]
        A[2 * i + 1][4] = y_des[i]
        A[2 * i + 1][5] = 1
        A[2 * i + 1][6] = -x_des[i] * y_src[i]
        A[2 * i + 1][7] = -y_des[i] * y_src[i]
        B[2 * i + 1][0] = y_src[i]

    return A, B

def fullMatrix(matchpoints):

    num = len(matchpoints)  # num is the number of salient key points
    matcher_A = np.ones((3, num))
    matcher_B = np.ones((3, num))
    number = 0

    for line in matchpoints:
        matcher_A[0][number] = line[1]  # col
        matcher_A[1][number] = line[0]  # row
        matcher_B[0][number] = line[3]  # col
        matcher_B[1][number] = line[2]  # row
        number += 1

    full_Matrix, delMatrix = toSolveFullMatrix(matcher_B, matcher_A, num)

    assert matcher_B.shape[1] == len(matchpoints)
    assert full_Matrix.shape[0] == delMatrix.shape[0]
    assert full_Matrix.shape[0] == 2 * len(matchpoints)
    return full_Matrix, delMatrix, matcher_B

def local_Hom(full_Matrix, delMatrix, cellCenterCoor, keyPoints):
    # cellCenterCoor = [col, row]
    # matcher_A: (3, points) [col, row, 1]
    # full_Matrix: (2*points, 8)
    # delMatrix: (2*points, 1)

    assert full_Matrix.shape[0] == 2 * keyPoints.shape[1]
    sigma, gamma = 8.5, 0.05

    dist = []
    for i in range(keyPoints.shape[1]):
        temp = (keyPoints[0][i] - cellCenterCoor[0])**2 + (keyPoints[1][i] - cellCenterCoor[1])**2
        temp = np.sqrt(temp)
        dist.append(temp)  # store the L2 distance between current grid cell center and all key-points

    dist = np.sort(dist)  # select the distance threshold
    distNumber = 7  # the selected key-points number
    flag = dist[distNumber]  # to ensure the flag distance of eight closet key-points

    for i in range(keyPoints.shape[1]):  # size (3, npoints)
        pdist2 = (keyPoints[0][i] - cellCenterCoor[0])**2 + (keyPoints[1][i] - cellCenterCoor[1])**2

        Gki = exp(-np.sqrt(pdist2)/(sigma ** 2))
        print("pdist2 is : ", np.sqrt(pdist2), Gki)
        # Wi = max(gamma, Gki)
        # Wi = Gki # weight parameter
        Wi = Gki if pdist2 < flag else 0.01  # select the distNumber closed key-points

        full_Matrix[2 * i] *= Wi
        full_Matrix[2 * i + 1] *= Wi
        delMatrix[2 * i] *= Wi
        delMatrix[2 * i + 1] *= Wi

    flag = np.linalg.inv(np.dot(full_Matrix.T, full_Matrix))
    flag1 = np.dot(flag, full_Matrix.T)  # MLS algorithm to solve local Hom

    h = np.dot(flag1, delMatrix)

    assert h.shape[0] == 8

    ans = np.ones((3, 3))
    ans[0][0] = h[0][0]
    ans[0][1] = h[1][0]
    ans[0][2] = h[2][0]
    ans[1][0] = h[3][0]
    ans[1][1] = h[4][0]
    ans[1][2] = h[5][0]
    ans[2][0] = h[6][0]
    ans[2][1] = h[7][0]
    ans[2][2] = 1

    return ans  # the local Hom of current grid cell

def local_Norm(imgASource, imgBSource, matcher_A, matcher_B, full_Matrix, delMatrix, keyPoints):

    # matcher_A, matcher_B : the points to be tested
    # keyPoints : the key features set in imgBSource

    dict = {}  # store the global matrix of each grid cell
    ##  from the right image B to the left image A
    error2 = 0
    num = 0
    num1 = 0
    cell = 100
    totalRow, totalCol = imgBSource.shape[0], imgBSource.shape[1]
    cellRow, cellCol = totalRow / cell, totalCol / cell  # the row and col length of each grid cell
    for j in range(matcher_B.shape[1]):
        cellNum = floor(matcher_B[0][j] / cellCol) + floor(matcher_B[1][j] / cellRow) * cell  # the grid cell number

        if cellNum not in dict:
            full, delM = full_Matrix.copy(), delMatrix.copy()
            centerCol = floor(matcher_B[0][j] / cellCol) * cellCol + cellCol / 2
            centerRow = floor(matcher_B[1][j] / cellRow) * cellRow + cellRow / 2
            cellCenterCoor = [centerCol, centerRow]  # [col, row]  the center coordinate of grid cell
            dict[cellNum] = local_Hom(full, delM, cellCenterCoor, keyPoints)

        globalHomMatrix = dict[cellNum]  # the local Hom of current grid cell

        matcher_B_after = np.dot(globalHomMatrix, matcher_B[:, j])
        matcher_B_after[0] = matcher_B_after[0] / matcher_B_after[2]
        matcher_B_after[1] = matcher_B_after[1] / matcher_B_after[2]
        matcher_B_after[2] = matcher_B_after[2] / matcher_B_after[2]

        # print("cost is :", globalHomMatrix, matcher_B[:, j], matcher_B_after, matcher_A[:, j])

        error = np.sqrt((matcher_A[0][j] - matcher_B_after[0]) ** 2 + (matcher_A[1][j] - matcher_B_after[1]) ** 2)
        if error < 0.1 * max(imgASource.shape[0], imgASource.shape[1]):
            num += 1
        if error < 0.05 * max(imgASource.shape[0], imgASource.shape[1]):
            num1 += 1

        ###############ERROR-LOC################
        error2 += (abs(matcher_A[0][j] - matcher_B_after[0]) / imgASource.shape[1] + \
                  abs(matcher_A[1][j] - matcher_B_after[1]) / imgASource.shape[0])

    return num, num1, error2