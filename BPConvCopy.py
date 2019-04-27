# CopyRight by BUAA, VR-Lab
# Author : Chen Lang
# 2018/08/02


from numpy import *
import numpy as np
# from numba import jit
from get_descriptor import *
import matplotlib.pyplot as plt
from numba import jit

posLow = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), \
          (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), \
          (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), \
          (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), \
          (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)]

posHigh = [(-1, -1), (-1, 0), (-1, 1), \
           (0, -1), (0, 0), (0, 1), \
           (1, -1), (1, 0), (1, 1)]
color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
eps = 0.0001

beta = 0.35  # 0.35
threshold = 0.55  # 0.4,0.7
def color_map(i):
    colors = [
        [255,0,0],
        [0,255,0],
        [0,0,255],
        [128,128,0],
        [0,128,128]
    ]

    if i < 5:
        return colors[i]
    else:
        return np.random.randint(0, 256, 3)

def extractPatch(coor, featureMap):
    # coor:list size is K * 4, stands for the leftUP and rightBottom Corner of every patch to search
    # featureMap: array size is row * col * Channel

    patch = list()
    for item in coor:
        patch.append(featureMap[item[0]:item[2]+1, item[1]:item[3]+1, :])
    return patch

def appendImage(imLeft, imRight):
    # select the image with the fewest rows and fill in enough empty rows
    # rowsNew = max(imLeft.shape[0], imRight.shape[0])
    # colNew = max(imLeft.shape[1], imRight.shape[1])

    rows1 = imLeft.shape[0]
    rows2 = imRight.shape[0]

    if len(imLeft.shape) == 3:
        assert imLeft.shape[2] == imRight.shape[2]
        depth = imLeft.shape[2]
        if rows1 < rows2:
            pad = ones((rows2-rows1, imLeft.shape[1], depth))*255
            imLeft = concatenate([imLeft, pad], axis=0).astype(uint8)  # 0为行扩展
        elif rows1 > rows2:
            pad = ones((rows1 - rows2, imRight.shape[1], depth))*255
            imRight = concatenate([imRight, pad], axis=0).astype(uint8)  # 数据类型转换要用arrayName.astype(dtype)
    elif len(imLeft.shape) == 2:
        if rows1 < rows2:
            pad = ones((rows2 - rows1, imLeft.shape[1])) * 255
            imLeft = concatenate([imLeft, pad], axis=0).astype(uint8)  # 0为行扩展
        elif rows1 > rows2:
            pad = ones((rows1 - rows2, imRight.shape[1])) * 255
            imRight = concatenate([imRight, pad], axis=0).astype(uint8)  # 数据类型转换要用arrayName.astype(dtype)

    imFinal = concatenate((imLeft, imRight), axis=1).astype(uint8)  # 列扩展
    # plt.imshow(imFinal)
    # plt.axis('off')
    # plt.show()
    return imFinal

def pyramidPlot(upFilter, reluFiveLeft, reluFiveRight, step):

    imgTotalSource = appendImage(reluFiveLeft[:, :, 0], reluFiveRight[:, :, 0])
    plt.imshow(imgTotalSource)

    flag = 0
    for item in upFilter:
        plt.plot(int(item[1]), int(item[0]), 'ro')
        plt.plot(int(item[3]) + step+1, int(item[2]), 'ro')
        plt.plot([int(item[1]), int(item[3]) + step+1], [int(item[0]), int(item[2])], color[flag % 8])
        flag += 1

    plt.colorbar()
    plt.axis('off')
    # plt.show()


def pyramidFiveRelu(reluFiveLeft, reluFiveRight):

    # reluFiveLeft: type is numpy.array and the value is relu5_1, size is row * col * Channel
    # reluFiveRight: type is numpy.array and the value is relu5_1, size is row1 * col1 * Channel
    # return is the matching local area in pyramid Four

    assert reluFiveLeft.shape[2] == reluFiveRight.shape[2]
    #####################print the begin information#############
    print("pyramidFiveRelu:\n")
    print("reluFiveLeft shape is :{}\n".format(reluFiveLeft.shape))
    print("reluFiveRight shape is :{}\n".format(reluFiveRight.shape))
    #####################print the begin information#############

    meanA, meanB = np.zeros((1, reluFiveLeft.shape[2])).squeeze(), np.zeros((1, reluFiveRight.shape[2])).squeeze()
    stdA, stdB = np.zeros((1, reluFiveLeft.shape[2])).squeeze(), np.zeros((1, reluFiveRight.shape[2])).squeeze()
    for i in range(reluFiveLeft.shape[2]):
        meanA[i], stdA[i] = reluFiveLeft[:, :, i].mean() + eps, reluFiveLeft[:, :, i].std() + eps
        meanB[i], stdB[i] = reluFiveRight[:, :, i].mean() + eps, reluFiveRight[:, :, i].std() + eps

    meanPQ, stdPQ = (meanA + meanB) / 2, (stdA + stdB) / 2
    # print(meanA, stdA)
    '''
    CaPQ = reluFiveLeft  # ((reluFiveLeft - meanA) / stdA) * stdPQ + meanPQ  # size is row * col * Channel
    CbQP = reluFiveRight  # ((reluFiveRight - meanB) / stdB) * stdPQ + meanPQ  # with size (row, col, Channel)
    '''
    CaPQ = ((reluFiveLeft - meanA) / stdA) * stdPQ + meanPQ  # size is row * col * Channel
    CbQP = ((reluFiveRight - meanB) / stdB) * stdPQ + meanPQ  # with size (row, col, Channel)

    assert meanPQ.shape == meanA.shape and stdPQ.shape == meanB.shape  # with size (1, Channel)
    assert CaPQ.shape == reluFiveLeft.shape and CbQP.shape == reluFiveRight.shape

    level = 1
    pad = 0
    # CaPQ, CbQP = np.pad(CaPQ, ((1,1),(1,1),(0,0)), 'constant'), np.pad(CbQP, ((1,1),(1,1),(0,0)), 'constant')

    ANNTable = np.zeros((reluFiveLeft.shape[0], reluFiveLeft.shape[1]))  # 存储A图区域中每个节点最近的B节点序号
    L2A = np.zeros((reluFiveLeft.shape[0], reluFiveLeft.shape[1]))
    L2B = np.zeros((reluFiveRight.shape[0], reluFiveRight.shape[1]))
    for i in range(0, reluFiveLeft.shape[0], 1):  # row
        for j in range(0, reluFiveLeft.shape[1], 1):  # col
            L2A[i][j] = sqrt(sum(reluFiveLeft[i][j]**2))
            L2B[i][j] = np.sqrt(sum(reluFiveRight[i][j] ** 2))

    HA = (L2A - np.min(L2A))/(np.max(L2A)-np.min(L2A))
    HB = (L2B - np.min(L2B))/(np.max(L2B)-np.min(L2B))

    for i in range(0, reluFiveLeft.shape[0], 1):  # row
        for j in range(0, reluFiveLeft.shape[1], 1):  # col
            if HA[i][j] < beta:
                ANNTable[i][j] = -1
                continue
            scoreTable = np.zeros((reluFiveRight.shape[0], reluFiveRight.shape[1]))  # 为每个A图节点建立一个B图的打分表
            for ii in range(0, reluFiveRight.shape[0], 1):
                for jj in range(0, reluFiveRight.shape[1], 1):
                    for dPos in posHigh:
                        if (i+dPos[0])<0 or (ii+dPos[0])<0 or (i+dPos[0])>= reluFiveLeft.shape[0] or (j+dPos[1])>=reluFiveLeft.shape[1] \
                                or (j+dPos[1])<0 or (jj+dPos[1])<0 or (ii+dPos[0])>= reluFiveRight.shape[0] or (jj+dPos[1])>=reluFiveRight.shape[1]:
                            continue
                        else:
                            score = sum(CaPQ[i+dPos[0]][j+dPos[1]] * CbQP[ii+dPos[0]][jj+dPos[1]])\
                            / (sqrt(sum(CbQP[ii+dPos[0]][jj+dPos[1]])**2)*sqrt(sum(CaPQ[i+dPos[0]][j+dPos[1]]**2)))

                            scoreTable[ii][jj] += score
            ANNTable[i][j] = np.argmax(scoreTable)  # A图当前节点在B中的搜索最佳位置

    BNNTable = np.zeros((reluFiveRight.shape[0], reluFiveRight.shape[1]))  # 存储B图区域中每个节点最近的A节点序号
    for i in range(0, reluFiveRight.shape[0], 1):  # row
        for j in range(0, reluFiveRight.shape[1], 1):  # col
            if HB[i][j] < beta:
                BNNTable[i][j] = -1
                continue

            scoreTable = np.zeros((reluFiveLeft.shape[0], reluFiveLeft.shape[1]))  # 为每个B图节点建立一个A图的打分表
            for ii in range(0, reluFiveLeft.shape[0],1):
                for jj in range(0, reluFiveLeft.shape[1],1):
                    for dPos in posHigh:
                        if (i + dPos[0]) < 0 or (ii + dPos[0]) < 0 or (ii + dPos[0]) >= reluFiveLeft.shape[0] or (jj + dPos[1]) >= reluFiveLeft.shape[1] \
                                or (j + dPos[1]) < 0 or (jj + dPos[1]) < 0 or (i + dPos[0]) >= reluFiveRight.shape[0] or (j + dPos[1]) >= reluFiveRight.shape[1]:
                            continue
                        else:
                            score = sum(CbQP[i+dPos[0]][j+dPos[1]] * CaPQ[ii+dPos[0]][jj+dPos[1]])\
                            / (np.sqrt(sum(CbQP[i+dPos[0]][j+dPos[1]])**2) * np.sqrt(sum(CaPQ[ii+dPos[0]][jj+dPos[1]]**2)))

                            scoreTable[ii][jj] += score
            BNNTable[i][j] = np.argmax(scoreTable)  # B图当前节点在A中搜索最佳位置

    upFive = list()
    for i in range(ANNTable.shape[0]):
        for j in range(ANNTable.shape[1]):
            if ANNTable[i][j] > -1:
                rowB, colB = divmod(ANNTable[i][j], reluFiveRight.shape[1])
                # print(rowB, colB)
                rowA, colA = divmod(BNNTable[int(rowB)][int(colB)], reluFiveLeft.shape[1])
                if rowA < 0 or colA < 0:
                    continue
                rowA, colA = int(rowA), int(colA)
                rowB, colB = int(rowB), int(colB)
                if rowA == i and colA == j:
                    ans = [rowA, colA, rowB, colB]  # find a corresponding region
                    upFive.append(ans)
            else:
                continue

    ##########################filter##############################
    # print(HA.max(),HA.min(),HB.min(),HB.max())
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(HA)
    # plt.subplot(1,2,2)
    # plt.imshow(HB)
    # plt.show()
    upFiveFilter = list()
    for item in upFive:
        if HA[item[0]][item[1]] > beta and HB[item[2]][item[3]] > beta and (item not in upFiveFilter):
            upFiveFilter.append(item)  # find a corresponding region which fits the threshold
        else:
            continue
    ##########################filter##############################


    #####################print the end information################
    print("pyramid " + str(5) + " end !\n")
    print("The number of NBB matching points  is : {}\n".format(len(upFiveFilter)))
    print("The correspondence regions in Layer_4 is : {}\n".format(len(upFiveFilter)))

    return upFiveFilter
    #####################print the end information################

def draw_square(image, center, color, radius = 2):
    d = 2*radius + 1
    image_p = np.pad(image, ((radius,radius),(radius,radius),(0,0)),'constant')
    center_p = [center[0]+radius, center[1]+radius]
    image_p[center_p[0]-radius, (center_p[1]-radius):(center_p[1]-radius+d), :] = np.tile(color,[d,1])
    image_p[(center_p[0]-radius):(center_p[0]-radius+d), center_p[1]-radius, :] = np.tile(color,[d,1])
    image_p[center_p[0]+radius, (center_p[1]-radius):(center_p[1]-radius+d), :] = np.tile(color,[d,1])
    image_p[(center_p[0]-radius):(center_p[0]-radius+d), center_p[1]+radius, :] = np.tile(color,[d,1])

    return image_p[radius:image_p.shape[0]-radius, radius:image_p.shape[1]-radius, :]

def drawCorrespondence(imgA, imgB, upFiveFilter, radius):
    # [rowA, colA, rowB, colB]
    A_marked, B_marked = imgA.copy(), imgB.copy()
    scale = 16
    flag = 0
    for i in range(len(upFiveFilter)):
        color = color_map(i)
        center_1 = [upFiveFilter[i][z] * scale for z in range(2)]
        center_2 = [upFiveFilter[i][z] * scale for z in range(2, 4)]
        A_marked = draw_square(A_marked, [center_1[0] + radius, center_1[1] + radius], color, radius=radius)
        B_marked = draw_square(B_marked, [center_2[0] + radius, center_2[1] + radius], color, radius=radius)
        flag += 1

    plt.figure(), plt.imshow(A_marked)
    plt.figure(), plt.imshow(B_marked)
    plt.axis('off')


    #################################
    # imgTotal = appendImage(imgA, imgB)
    batch = np.concatenate((imgA, imgB), 1)
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
    flag = 0
    plt.figure(), plt.imshow(batch)
    plt.axis('off')

    for i in range(len(upFiveFilter)):
        center_1 = [upFiveFilter[i][z] * scale for z in range(2)]
        center_2 = [upFiveFilter[i][z] * scale for z in range(2, 4)]
        plt.plot(center_1[1], center_1[0], color[flag % 8] + '.')  # show points
        plt.plot(224 + center_2[1], center_2[0], color[flag % 8] + '.')  # show points
        plt.plot([center_1[1], 224 + center_2[1]], [center_1[0], center_2[0]], color[flag % 8])  # True matching
        flag += 1
    #################################

    plt.show()

def pyramidFive(reluFiveLeft, reluFiveRight):

    upFiveFilter = pyramidFiveRelu(reluFiveLeft, reluFiveRight)
    matchFourPatch = list()

    # drawCorrespondence(imgA, imgB, upFiveFilter, 8)

    # upFiveFilter = pyramidRansac(upFiveFilter, reluFiveLeft, reluFiveRight, 14, 1.0)  # use ransac to filter out some points

    for item in upFiveFilter:

        #########################the Patch in A########################
        LowApx, highApx = 2*item[0] - 3, 2*item[0] + 3
        LowApy, highApy = 2*item[1] - 3, 2*item[1] + 3
        LowBpx, highBpx = 2*item[2] - 3, 2*item[2] + 3
        LowBpy, highBpy = 2*item[3] - 3, 2*item[3] + 3
        matchFourPatch.append([LowApx, LowApy, highApx, highApy, LowBpx, LowBpy, highBpx, highBpy])  # lowRow,lowCol,highRow,highCol
        #########################the Patch in A########################

    return upFiveFilter, matchFourPatch

@jit
def pyramidRelu(item, convLeft, convRight, level, leftFlag, rightFlag):
    # item = [LowApx, LowApy, highApx, highApy, LowBpx, LowBpy, highBpx, highBpy]
    # convLeft: type is np.array with size (row, col, Channel), means the relu_1 of pyramid_level
    # convRight: type is np.array with size (row1, col1, Channel), means the relu_1 of prramid_level
    # print(item)
    # print("{} layer is {}*{}".format(level, item[2]-item[0], item[3]-item[1]))
    assert len(item) == 8
    assert convLeft.shape[2] == convRight.shape[2]
    assert (item[2]-item[0]) == (item[6]-item[4]) and (item[3]-item[1]) == (item[7]-item[5])

    #####################print the begin information#############
    # print("pyramid" + str(level) + "Relu:\n")
    # print("reluFiveLeft shape is :{}\n".format(convLeft.shape))
    # print("reluFiveRight shape is :{}\n".format(convRight.shape))
    #####################print the begin information#############

    meanA, stdA = np.zeros((1, convLeft.shape[2])).squeeze(), np.zeros((1, convLeft.shape[2])).squeeze()
    meanB, stdB = np.zeros((1, convRight.shape[2])).squeeze(), np.zeros((1, convRight.shape[2])).squeeze()
    for i in range(convLeft.shape[2]):
        # print(convLeft[item[0]:(item[2]+1), item[1]:(item[3]+1), i])
        meanA[i] = convLeft[item[0]:(item[2]+1), item[1]:(item[3]+1), i].mean() + eps
        stdA[i] = convLeft[item[0]:1+item[2], item[1]:1+item[3], i].std() + eps
        meanB[i] = convRight[item[4]:1+item[6], item[5]:1+item[7], i].mean() + eps
        stdB[i] = convRight[item[4]:1+item[6], item[5]:1+item[7], i].std() + eps

    # print(meanA)
    meanPQ, stdPQ = (meanA + meanB) / 2, (stdA + stdB) / 2  # 5 * 5 * Channel or 7 * 7 * Channel
    CaPQ = ((convLeft - meanA) / stdA) * stdPQ + meanPQ
    CbQP = ((convRight - meanB) / stdB) * stdPQ + meanPQ
    assert CaPQ.shape == convLeft.shape and CbQP.shape == convRight.shape
    assert meanPQ.shape == stdPQ.shape and meanA.shape == meanB.shape

    pos = posHigh if level >= 3 else posLow

    # r = 1
    # pad = 0 if level >= 4 else 2
    # item = [LowApx, LowApy, highApx, highApy, LowBpx, LowBpy, highBpx, highBpy]
    ANNTable = np.zeros((item[2]-item[0]+1, item[3]-item[1]+1))
    L2A = np.zeros((item[2]-item[0]+1, item[3]-item[1]+1))
    L2B = np.zeros((item[6] - item[4] + 1, item[7] - item[5] + 1))
    # starttime = datetime.datetime.now()
    for i in range(item[0], item[2] + 1, 1):
        for j in range(item[1], item[3] + 1, 1):
            L2A[i-item[0]][j-item[1]] = np.sqrt(sum(convLeft[i][j]**2))
            L2B[i-item[0]][j-item[1]] = np.sqrt(sum(convRight[i-item[0]+item[4]][j-item[1]+item[5]] ** 2))
    HaP = (L2A - L2A.min())/(L2A.max()-L2A.min())
    HbQ = (L2B - L2B.min())/(L2B.max()-L2B.min())

    for i in range(item[0], item[2]+1, 1):
        for j in range(item[1], item[3]+1, 1):
            # if leftFlag[i][j]:
            #     ANNTable[i - item[0]][j - item[1]] = -1
            #     continue

            leftFlag[i][j] = 1
            if HaP[i-item[0]][j-item[1]] < threshold:
                ANNTable[i - item[0]][j - item[1]] = -1
                continue
            scoreTable = np.zeros((item[6]-item[4]+1, item[7]-item[5]+1))
            for ii in range(item[4], item[6]+1, 1):
                for jj in range(item[5], item[7]+1, 1):
                    for dPos in pos:
                        if (i + dPos[0]) < 0 or (ii + dPos[0]) < 0 or (i + dPos[0]) >= convLeft.shape[0] or (j + dPos[1]) >= convLeft.shape[1] \
                                or (j + dPos[1]) < 0 or (jj + dPos[1]) < 0 or (ii + dPos[0]) >= convRight.shape[0] or (jj + dPos[1]) >= convRight.shape[1]:
                            continue
                        else:
                            score = sum(CaPQ[i+dPos[0]][j+dPos[1]]*CbQP[ii+dPos[0]][jj+dPos[1]])/(\
                                np.sqrt(sum(CaPQ[i+dPos[0]][j+dPos[1]]**2)*sum(CbQP[ii+dPos[0]][jj+dPos[1]]**2)))
                            scoreTable[ii-item[4]][jj-item[5]] += score

            ANNTable[i-item[0]][j-item[1]] = np.argmax(scoreTable)  # 存储的是相对(item[0], item[1])的相对位置序号

    # endtime1 = datetime.datetime.now()
    # print("nns is :",endtime1-endtime)

    BNNTable = np.zeros((item[6] - item[4] + 1, item[7] - item[5] + 1))
    # L2B = np.zeros((item[6] - item[4] + 1, item[7] - item[5] + 1))
    for i in range(item[4], item[6] + 1, 1):
        for j in range(item[5], item[7] + 1, 1):
            # if rightFlag[i][j]:
            #     BNNTable[i - item[4]][j - item[5]] = -1
            #     continue

            rightFlag[i][j] = 1
            if HbQ[i-item[4]][j-item[5]] < threshold:
                BNNTable[i - item[4]][j - item[5]] = -1
                continue
            scoreTable = np.zeros((item[2] - item[0] + 1, item[3] - item[1] + 1))
            for ii in range(item[0], item[2] + 1, 1):
                for jj in range(item[1], item[3] + 1, 1):
                    for dPos in pos:
                        if (i + dPos[0]) < 0 or (ii + dPos[0]) < 0 or (ii + dPos[0]) >= convLeft.shape[0] or (jj + dPos[1]) >= convLeft.shape[1] \
                                or (j + dPos[1]) < 0 or (jj + dPos[1]) < 0 or (i + dPos[0]) >= convRight.shape[0] or (j + dPos[1]) >= convRight.shape[1]:
                            continue
                        else:
                            score = sum(CbQP[i + dPos[0]][j + dPos[1]] * CaPQ[ii + dPos[0]][jj + dPos[1]]) / ( \
                                np.sqrt(sum(CbQP[i+dPos[0]][j+dPos[1]]**2) * sum(CaPQ[ii+dPos[0]][jj+dPos[1]]**2)))
                            scoreTable[ii-item[0]][jj-item[1]] += score

            BNNTable[i-item[4]][j-item[5]] = np.argmax(scoreTable)  # 存储的是相对(item[4], item[5])的相对位置序号
            # L2B[i - item[4]][j - item[5]] = np.sqrt(sum(convRight[i][j] ** 2))

    upCoor = list()
    for i in range(ANNTable.shape[0]):
        for j in range(ANNTable.shape[1]):
            if ANNTable[i][j] > -1:
                rowB, colB = divmod(ANNTable[i][j], item[7]-item[5]+1)
                rowA, colA = divmod(BNNTable[int(rowB)][int(colB)], item[3]-item[1]+1)
                if rowA < 0 or colA < 0:
                    continue
                rowA, colA = rowA + item[0], colA + item[1]
                rowB, colB = rowB + item[4], colB + item[5]
                rowA, colA = int(rowA), int(colA)
                rowB, colB = int(rowB), int(colB)

                if rowA == (i+item[0]) and colA == (j+item[1]):
                    upCoor.append([rowA, colA, rowB, colB])
            else:
                continue

    ##########################filter##############################
    upFilter = list()
    # meanH, meanL = (L2A.max() + L2B.max()) / 2, (L2A.min() + L2B.min()) / 2
    # beta = meanL + 0.6 * (meanH - meanL)
    #  0.5 used for source and 0.4 used for cropped image
    '''
    HaP = (L2A - L2A.min())/(L2A.max()-L2A.min())
    HbQ = (L2B - L2B.min())/(L2B.max()-L2A.min())

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(HaP)
    # plt.subplot(1,2,2)
    # plt.imshow(HbQ)
    # plt.show()

    for ele in upCoor:
        if HaP[ele[0]-item[0]][ele[1]-item[1]] > threshold and HbQ[ele[2]-item[4]][ele[3]-item[5]] > threshold:
            upFilter.append(ele)
        else:
            continue
    ##########################filter##############################

    #####################print the end information################
    '''
    static = 8 if level >=3 else 5 if level >=2 else 2
    if len(upCoor) > static:
        ans = []
        alist = list(np.random.randint(0, len(upCoor), static))
        for item in alist:
            ans.append(upCoor[item])
        upCoor = ans
    # print("cur return size is:", len(upCoor))
    return upCoor, leftFlag, rightFlag
    #####################print the end information################

@jit
def pyramidLevel(matchtPatchCoor, convLeft, convRight, level):
    # matchtFourPatchCoor: list size is K * 8, K is the number of the patch to search in feature map of level four
    #                      8 means the flaged coordinates of matching patch to search in two feature map
    # convLeft: array size is row * col * Channel, is the left feature map
    # convRight:array with size row * col * Channel, is the right feature map

    upThree = list()
    convLeftFlag = np.zeros((convLeft.shape[0], convLeft.shape[1]))
    convRightFlag = np.zeros((convRight.shape[0], convRight.shape[1]))
    for item in matchtPatchCoor:
        if item[0] < 0 or item[1] < 0 or item[4] < 0 or item[5] < 0 or item[2] >= convLeft.shape[0] or item[3] >= \
                convLeft.shape[1] or item[6] >= convRight.shape[0] or item[7] >= convRight.shape[1]:
            continue
        else:
            currentPatchFilter, convLeftFlag, convRightFlag = pyramidRelu(item, convLeft, convRight, level, convLeftFlag, convRightFlag)
            for ele in currentPatchFilter:
                if ele not in upThree:
                    upThree.append(ele)
                else:
                    continue

    # upThree = pyramidRansac(upThree, convLeft, convRight, 14*(2**(5-level)), 4**(5-level))
    # pyramidPlot(upThree, convLeft, convRight, 14 * (2 ** (5 - level)))

    if level == 2 or level == 3:
       upThree = pyramidRansac(upThree, convLeft, convRight, 14 * (2 ** (5 - level)), 200)

    matchtUpPatchCoor = list()
    # [LowApx, LowApy, highApx, highApy, LowBpx, LowBpy, highBpx, highBpy]
    radius = 3 if level >= 4 else 2
    if level == 1:
        return upThree
    for item in upThree:
        # item: [Apx, Apy, Bpx, Bpy]
        LowApx, HighApx = 2 * item[0] - radius, 2 * item[0] + radius
        LowApy, HighApy = 2 * item[1] - radius, 2 * item[1] + radius
        LowBpx, HighBpx = 2 * item[2] - radius, 2 * item[2] + radius
        LowBpy, HighBpy = 2 * item[3] - radius, 2 * item[3] + radius
        matchtUpPatchCoor.append([LowApx, LowApy, HighApx, HighApy, LowBpx, LowBpy, HighBpx, HighBpy])

    print("pyramid " + str(level) + "  end !\n")
    print("The number of NBB matching points  is : {}\n".format(len(matchtUpPatchCoor)))
    print("The correspondence regions in Layer_" + str(level-1) + " (before filtering) is : {}\n".format(len(matchtUpPatchCoor)))

    return upThree, matchtUpPatchCoor

def writeFilename(matchPoints, filename):
    with open(filename, 'a') as f:
        for item in matchPoints:
            f.write(str(item[0]) + ',' + str(item[1]) + ',' + str(item[2]) + ',' + str(item[3]) + '\n')

def plotSameStyle(A, B):
    assert A.shape[2] == B.shape[2]

    meanA, meanB = np.zeros((1, A.shape[2])).squeeze(), np.zeros((1, B.shape[2])).squeeze()
    stdA, stdB = np.zeros((1, A.shape[2])).squeeze(), np.zeros((1, B.shape[2])).squeeze()
    for i in range(A.shape[2]):
        meanA[i], stdA[i] = A[:, :, i].mean() + eps, A[:, :, i].std() + eps
        meanB[i], stdB[i] = B[:, :, i].mean() + eps, B[:, :, i].std() + eps

    meanPQ, stdPQ = (meanA + meanB) / 2, (stdA + stdB) / 2
    # print(meanA, stdA)
    CaPQ = ((A - meanA) / stdA) * stdPQ + meanPQ  # size is row * col * Channel
    CbQP = ((B - meanB) / stdB) * stdPQ + meanPQ  # with size (row, col, Channel)
    appendImage(CaPQ, CbQP)  # plot the inter-same style image

def pyramidRansac(upFilter, reluFiveLeft, reluFiveRight, step, err):
    imgTotalSource = appendImage(reluFiveLeft[:, :,0], reluFiveRight[:, :, 0])
    # plt.imshow(imgTotalSource)
    # flag = 0
    # for item in upFilter:
    #     plt.plot(int(item[1]), int(item[0]), 'ro')
    #     plt.plot(int(item[3]) + step, int(item[2]), 'ro')
    #     plt.plot([int(item[1]), int(item[3]) + step], [int(item[0]), int(item[2])], color[flag % 8])
    #     flag += 1
    #
    # plt.colorbar()
    # plt.axis('off')
    # plt.show()

    upFiveFilter, lenth, pointsNum, rightMatrix = plot_ransac_match(upFilter, step, imgTotalSource, err, [1, 1, 1, 1])

    return upFiveFilter

def plot_Source(matchPoints, imgTotalSource, depth, imgASource, imgBSource):
    # filename = 'NBB.txt'
    # writeFilename(matchPoints, filename)
    scale0A, scale1A = imgASource.shape[0] / depth, imgASource.shape[1] / depth  # used for plot the source scale imageA
    scale0B, scale1B = imgBSource.shape[0] / depth, imgBSource.shape[1] / depth  # used for plot the source scale imageB
    # plt.title("NBB with " + str(len(matchPoints)) + 'points')
    plt.title(str(depth))
    plt.imshow(imgTotalSource)
    flag = 0
    color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
    for item in matchPoints:
        plt.plot(int(item[1] * scale1A), int(item[0] * scale0A), 'r.')
        plt.plot(int(item[3] * scale1B) + imgASource.shape[1], int(item[2] * scale0B), 'r.')
        plt.plot([int(item[1] * scale1A), int(item[3] * scale1B) + imgASource.shape[1]], \
                 [int(item[0] * scale0A), int(item[2] * scale0B)], color[flag % 8])
        item[1] = int(item[1] * scale1A)
        item[0] = int(item[0] * scale0A)
        item[3] = int(item[3] * scale1B)
        item[2] = int(item[2] * scale0B)
        flag += 1

    plt.axis('off')
    plt.savefig(str(depth) + 'real.png', dpi=300)
    plt.show()