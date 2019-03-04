# CopyRight by BUAA, VR-Lab
# Author : Chen Lang
# 2018/08/02

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from BPConv import *
import vgg19
from utils import *
from get_descriptor import *
import xlrd, xlwt, cv2
from PCK import *
from global_Hom import *
from skimage import draw


def maskBool(rowCoor, colCool, img1Source):
    fill_row_coords, fill_col_coords = draw.polygon(rowCoor, colCool, (img1Source.shape[0], img1Source.shape[1]))  # (row, col)
    mask = np.zeros((img1Source.shape[0], img1Source.shape[1]), dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask + 0

def intersectionOverUnion(wapredMask, mask):
    intersection = np.sum(warpedMask & mask)
    union = np.sum(wapredMask | mask)
    IoU = intersection / union
    return IoU

def label_transfer_accuracy(wapredMask, mask):
    labelMask = (wapredMask == mask) + 0
    LT_ACC = np.mean(labelMask)
    return LT_ACC

def localization_error(wapredMask, mask, matcher_A, matcher_B, matrix):
    XoA, YoA = int(matcher_A[0].min()), int(matcher_A[1].min())
    XoB, YoB = int(matcher_B[0].min()), int(matcher_B[1].min())
    wA, hA = int(matcher_A[0].max() - matcher_A[0].min() + 1), int(matcher_A[1].max() - matcher_A[1].min() + 1)
    wB, hB = int(matcher_B[0].max() - matcher_B[0].min() + 1), int(matcher_B[1].max() - matcher_B[1].min() + 1)

    V = 0  # the total points in edge of source image
    error = 0

    for i in range(XoA, XoA + wA):  # col
        for j in range(YoA, YoA + hA):  # row
            iWarp = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2]
            jWarp = matrix[1][0] * i + matrix[1][1] * j + matrix[1][2]
            delWarp = matrix[2][0] * i + matrix[2][1] * j + matrix[2][2]
            iWarp = int(iWarp / delWarp)
            jWarp = int(jWarp / delWarp)
            if iWarp < 0 or jWarp < 0 or iWarp > mask.shape[1] or jWarp > mask.shape[0]:
                continue
            if iWarp >= XoB and iWarp <= matcher_B[0].max() and jWarp >= YoB and jWarp <= matcher_B[1].max():
                V += 1
                xA, yA = (i - XoA) / wA, (j - YoA) / hA
                xWarp, yWarp = (iWarp - XoB) / wB, (jWarp - YoB) / hB
                error += (abs(xA - xWarp) + abs(yA - yWarp))

    LOC_ERR = error / V if V else -1
    return LOC_ERR


if __name__ == '__main__':
    # file = './Vgg19Net.xlsx'
    file = './test_pairs_caltech_with_category.xlsx'
    xlfile = xlrd.open_workbook(file)

    sheet_name = xlfile.sheet_names()[0]

    sheet1 = xlfile.sheet_by_name(sheet_name)

    rows, cols = sheet1.nrows, sheet1.ncols
    sess = tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(allow_growth=True))))  # 动态申请内存空间
    images = tf.placeholder("float", [2, 224, 224, 3])
    fileName = "Caltech.txt"

    for i in range(1, rows, 3):
        src = sheet1.cell_value(i, 0)
        dst = sheet1.cell_value(i, 1)
        XA = sheet1.cell_value(i, 3).split(',')
        YA = sheet1.cell_value(i, 4).split(',')
        XB = sheet1.cell_value(i, 5).split(',')
        YB = sheet1.cell_value(i, 6).split(',')

        print(src)
        num = len(XA)
        matcher_A = np.ones((3, num))
        matcher_B = np.ones((3, len(XB)))

        for number in range(0, num):
            matcher_A[0][number] = float(XA[number])  # col-y
            matcher_A[1][number] = float(YA[number])  # row-x
        for number in range(len(XB)):
            matcher_B[0][number] = float(XB[number])
            matcher_B[1][number] = float(YB[number])

        path = "D:\Code_Image_Stitch\weakalign-master\datasets/101_ObjectCategories/"
        img1, img1Source = load_image(path + src)
        img2, img2Source = load_image(path + dst)
        print(img2Source.shape)
        # plt.figure(), plt.imshow(img1Source)
        # # for i in range(num):
        # #     plt.plot(matcher_A[0][i], matcher_A[1][i],'ro')
        maskA = maskBool(matcher_A[1], matcher_A[0], img1Source)
        maskB = maskBool(matcher_B[1], matcher_B[0], img2Source)
        # plt.figure(), plt.imshow(maskA, 'gray')
        # plt.figure(), plt.imshow(maskB, 'gray')
        # plt.show()

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

        Filter, lenth, pointsNum, rightMatrix = plot_ransac_match(matchPoints, imgASource.shape[1], imgTotalSource, 200, [1, 1, 1, 1])

        # rightMatrix = np.array([[1.21208655e+00, 5.76444904e-02 ,-1.05272851e+02],[-7.43513422e-02 ,1.30137172e+00, -8.18588840e+01],[-4.53348533e-04 ,8.21595950e-05, 1.00000000e+00]])
        mask3D = imgASource.copy()
        mask3D[:, :, 0] = maskA
        warpedAffine1 = cv2.warpPerspective(mask3D, np.linalg.inv(rightMatrix), (imgBSource.shape[1], imgBSource.shape[0]))  # newWarpAffine(img1, img2, Hom)
        warpedMask = warpedAffine1[:, :, 0]
        # plt.imshow(warpedMask, 'gray')
        # plt.title("warped mask")
        # plt.axis('off')
        # plt.show()
        LA_TCC = label_transfer_accuracy(warpedMask, maskB)
        IoU = intersectionOverUnion(warpedMask, maskB)
        LOC_ERR = localization_error(warpedMask, maskB, matcher_A, matcher_B, np.linalg.inv(rightMatrix))
        print(LA_TCC, IoU, LOC_ERR)
        with open(fileName, 'a') as f:  # the second img
            f.write(str(LA_TCC) + ' ,' + str(IoU) + ' ,' + str(LOC_ERR) + ' ,' + '\n')