import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from BPConv import *
import vgg19
from utils import *
from get_descriptor import *
import cv2
import datetime
from numba import jit

cur = 'D:/Code_Image_Stitch/Vgg19Net/test_data/demo/'
# img1, img1Source = load_image(cur + 'car/3.jpg')
# img2, img2Source = load_image(cur + 'car/4.jpg')
img1, img1Source = load_image("./test_data/bird/2008_001194.jpg")
img2, img2Source = load_image("./test_data/bird/2008_002429.jpg")

imgA, imgB = img1, img2  # (img1 * 255).astype(uint8), (img2*255).astype(uint8)
imgASource, imgBSource = img1Source[:, :, 0:3], img2Source[:, :, 0:3]
img1, img2 = img1[:, :, 0:3], img2[:, :, 0:3]

depth = 224  # 224
scale0A, scale1A = imgASource.shape[0] / depth, imgASource.shape[1] / depth  # used for plot the source scale imageA
scale0B, scale1B = imgBSource.shape[0] / depth, imgBSource.shape[1] / depth  # used for plot the source scale imageB
imgTotal = appendImage(imgA, imgB)
print(imgA.shape, imgB.shape, imgASource.shape, imgBSource.shape)
imgTotalSource = appendImage(imgASource, imgBSource)  # the source linked image
# plotSameStyle(imgASource, imgBSource)

shape = (1, 224, 224, 3)
batch1 = img1.reshape(shape)
batch2 = img2.reshape(shape)
# batch1 = img1.reshape((1, 224, 224, 3))
batch = np.concatenate((batch1, batch2), 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:

        images = tf.placeholder("float", [2, 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        convList = sess.run(vgg.convList, feed_dict=feed_dict)

        for i in range(len(convList)):
            print(convList[i].shape)
            plt.matshow(convList[i][1, :, :, 0], cmap=plt.cm.hot, fignum=i + 1)
            plt.colorbar()
            plt.show()  # to visualize the deep feature pyramid

        convLeft, convRight = list(), list()
        for item in convList:
            convLeft.append(item[0])
            convRight.append(item[1])  # get the five levels of relul_1 feature map

##########################################The TOP Layer#############################################
        starttime = datetime.datetime.now()
        matchFivePoints, matchtFourPatchCoor = pyramidFive(convLeft[0], convRight[0])
        plot_Source(matchFivePoints, imgTotalSource, 14, imgASource, imgBSource)
####################################################################################################
        matchFourPoints, matchtThreePatchCoor = pyramidLevel(matchtFourPatchCoor, convLeft[1], convRight[1], 4)
        plot_Source(matchFourPoints, imgTotalSource, 28, imgASource, imgBSource)
        # matchtThreePatchCoor = pyramidFive(convLeft[1], convRight[1])

        upThree, matchtTwoPatchCoor = pyramidLevel(matchtThreePatchCoor, convLeft[2], convRight[2], 3)
        plot_Source(upThree, imgTotalSource, 56, imgASource, imgBSource)

        endtime = datetime.datetime.now()
        print("running time is {} s".format((endtime-starttime).seconds))

        upTwo, matchtOnePatchCoor = pyramidLevel(matchtTwoPatchCoor, convLeft[3], convRight[3], 2)
        plot_Source(upTwo, imgTotalSource, 112, imgASource, imgBSource)

        matchPoints = pyramidLevel(matchtOnePatchCoor, convLeft[4], convRight[4], 1)

#######################################################################################################

        ##################################No Ransac Filter############################
        plt.imshow(imgTotalSource)
        flag = 0
        color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
        # plt.title("NBB with " + str(len(matchPoints)) + 'points')
        for item in matchPoints:
            plt.plot(int(item[1] * scale1A), int(item[0] * scale0A), 'ro')
            plt.plot(int(item[3] * scale1B) + imgASource.shape[1], int(item[2] * scale0B), 'ro')
            plt.plot([int(item[1] * scale1A), int(item[3] * scale1B) + imgASource.shape[1]], \
                     [int(item[0] * scale0A), int(item[2] * scale0B)], color[flag % 8])
            item[1] = int(item[1] * scale1A)
            item[0] = int(item[0] * scale0A)
            item[3] = int(item[3] * scale1B)
            item[2] = int(item[2] * scale0B)
            flag += 1
        plt.axis('off')
        plt.show()
        ###############################################################
        Filter, lenth, pointsNum, rightMatrix = plot_ransac_match(matchPoints, imgASource.shape[1], imgTotalSource, 200, \
                                             [1, 1, 1, 1])

        print("right matrix is :", rightMatrix)
        warpedAffine1 = cv2.warpPerspective(imgASource, np.linalg.inv(rightMatrix), (imgBSource.shape[1], imgBSource.shape[0]))  # newWarpAffine(img1, img2, Hom)
        plt.imshow(warpedAffine1)
        plt.axis('off')
        plt.show()
        # print("The number of coarse matching pairs is {}\n".format(lenth))
        # print("The final matching pairs is {}\n".format(pointsNum))
        # print("The final percentage is {}\n".format(pointsNum / lenth))


        '''
        ################################################## GLOBAL Homography ###########################################

        


        ################################################## GLOBAL Homography ###########################################
        '''