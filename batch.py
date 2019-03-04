import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from BPConv import *
import vgg19
from utils import *
from get_descriptor import *
import xlrd, xlwt
import numpy as np
from PCK import *
from global_Hom import *
import datetime

# file = './Vgg19Net.xlsx'
file = './test_pairs_pf_pascal.xlsx'
xlfile = xlrd.open_workbook(file)

sheet_name = xlfile.sheet_names()[0]

sheet1 = xlfile.sheet_by_name(sheet_name)

rows, cols = sheet1.nrows, sheet1.ncols
sess = tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(allow_growth=True))))  # 动态申请内存空间
images = tf.placeholder("float", [2, 224, 224, 3])
fileName = "result.txt"
fileName1 = "error.txt"
for i in range(1, rows):
    src = sheet1.cell_value(i, 0)
    dst = sheet1.cell_value(i, 1)
    XA = sheet1.cell_value(i, 3).split(';')
    YA = sheet1.cell_value(i, 4).split(';')
    XB = sheet1.cell_value(i, 5).split(';')
    YB = sheet1.cell_value(i, 6).split(';')
# for i in range(1, rows, 2):
#     src = sheet1.cell_value(i, 0)
#     dst = sheet1.cell_value(i, 1)
#     XA= list()
#     YA= list()
#     XB= list()
#     YB= list()
#     for j in range(2, 12):
#         XA.append(sheet1.cell_value(i, j))
#         YA.append(sheet1.cell_value(i, j+10))
#         XB.append(sheet1.cell_value(i, j+20))
#         YB.append(sheet1.cell_value(i, j+30))

    print(XA)
    num = len(XA)
    matcher_A = np.ones((3, num))
    matcher_B = np.ones((3, num))

    for number in range(0, num):
        matcher_A[0][number] = float(XA[number])  # col-y
        matcher_A[1][number] = float(YA[number])  # row-x
        matcher_B[0][number] = float(XB[number])
        matcher_B[1][number] = float(YB[number])

    print(src, dst)
    path = "D:\Code_Image_Stitch\Vgg19Net/" + src
    img1, img1Source = load_image(path)
    img2, img2Source = load_image("D:\Code_Image_Stitch\Vgg19Net/" + dst)

    imgA, imgB = img1, img2
    imgASource, imgBSource = img1Source[:, :, 0:3], img2Source[:, :, 0:3]
    img1, img2 = img1[:, :, 0:3], img2[:, :, 0:3]
    scale0A, scale1A = imgASource.shape[0] / 224, imgASource.shape[1] / 224  # used for plot the source scale imageA
    scale0B, scale1B = imgBSource.shape[0] / 224, imgBSource.shape[1] / 224  # used for plot the source scale imageB
    imgTotal = appendImage(imgA, imgB)
    print(imgA.shape, imgB.shape, imgASource.shape, imgBSource.shape)
    imgTotalSource = appendImage(imgASource, imgBSource)  # the source linked image
    # plotSameStyle(imgASource, imgBSource)

    shape = (1, 224, 224, 3)
    batch1 = img1.reshape(shape)
    batch2 = img2.reshape(shape)
    batch = np.concatenate((batch1, batch2), 0)

    # images = tf.placeholder("float", [2, 224, 224, 3])
    feed_dict = {images: batch}

    vgg = vgg19.Vgg19()
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    convList = sess.run(vgg.convList, feed_dict=feed_dict)

    convLeft, convRight = list(), list()
    for item in convList:
        convLeft.append(item[0])
        convRight.append(item[1])  # get the five levels of relul_1 feature map

    matchFivePoints, matchtFourPatchCoor = pyramidFive(convLeft[0], convRight[0])
    # plot_Source(matchFivePoints, imgTotalSource, 14, imgASource, imgBSource)

    # Filter = matchFivePoints
    matchFourPoints, matchtThreePatchCoor = pyramidLevel(matchtFourPatchCoor, convLeft[1], convRight[1], 4)

    # matchtThreePatchCoor = pyramidFive(convLeft[1], convRight[1])

    upThree, matchtTwoPatchCoor = pyramidLevel(matchtThreePatchCoor, convLeft[2], convRight[2], 3)

    upTwo, matchtOnePatchCoor = pyramidLevel(matchtTwoPatchCoor, convLeft[3], convRight[3], 2)
    # plot_Source(upTwo, imgTotalSource, 112, imgASource, imgBSource)

    matchPoints = pyramidLevel(matchtOnePatchCoor, convLeft[4], convRight[4], 1)

    for item in matchPoints:
        item[1] = int(item[1] * scale1A)  # col-y
        item[0] = int(item[0] * scale0A)  # row-x
        item[3] = int(item[3] * scale1B)
        item[2] = int(item[2] * scale0B)

    Filter, lenth, pointsNum, rightMatrix = plot_ransac_match(matchPoints, imgASource.shape[1], imgTotalSource, 200, \
                                         [1, 1, 1, 1])

    pckVal, pckVal2, lac = pck(imgASource, imgBSource, matcher_A, matcher_B, rightMatrix)

    ################################################## LOCAL Homography ###########################################
    full_Matrix, delMatrix, keyPoints = fullMatrix(Filter)
    num, num1, error2 = local_Norm(imgASource, imgBSource, matcher_A, matcher_B, full_Matrix, delMatrix, keyPoints)  # local_Hom
    ################################################## LOCAL Homography ###########################################

    with open(fileName, 'a') as f:  # the second img
        f.write(src + ',' + str(pckVal) + ',' + str(pckVal2) + ',' + str(num / matcher_B.shape[1]) + ',' + str(num1 / matcher_B.shape[1])+'\n')

    # print("error : ", error)
    with open(fileName1, 'a') as g:  # the second img
        g.write(src + ':' + str(lac/matcher_B.shape[1]) + ',' + str(error2/matcher_B.shape[1]) + '\n')