import scipy.io
import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from BPConv import *
import vgg19
import utils

def _conv_layer(input, weights, bias):
    # 由于此处使用的是已经训练好的VGG-19参数，所有weights可以定义为常量
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)


# 加载现成的VGG参数,使每一层的名字和对应的参数对应起来
def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    # 原始VGG中，对输入的数据进行了减均值的操作，借别人的参数使用时也需要进行此步骤
    # 获取每个通道的均值，打印输出每个通道的均值为[ 123.68   116.779  103.939]
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    # 定义net字典结构，key为层的名字，value保存每一层使用VGG参数运算后的结果
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # 注意：Mat中的weights参数和tensorflow中不同
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # mat weight.shape: (3, 3, 3, 64)
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            # 扁平化
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel, layers
# 致次完成一次前向传播


cwd = os.getcwd()
VGG_PATH = cwd + "/imagenet-vgg-verydeep-19.mat"
IMG_PATHA = cwd + "/test_data/car/2010_004059.jpg"
IMG_PATHB = cwd + "/test_data/bike/2009_004561.jpg"
input_image = imread(IMG_PATHA)
input_image = input_image[:, :, 0:3]
input_imageB = imread(IMG_PATHB)
input_imageB = input_imageB[:, :, 0:3]
shape = (1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
shapeB = (1, input_imageB.shape[0], input_imageB.shape[1], input_imageB.shape[2])
convLeft, convRight = list(), list()
with tf.Session() as sess:
    image = tf.placeholder('float', shape=shape)
    nets, mean_pixel, all_layers = net(VGG_PATH, image)
    input_image_pre = np.array([preprocess(input_image, mean_pixel)])
    layers = all_layers  # For all layers 
    for i, layer in enumerate(layers):
        #  feature:[batch数，H ,W ,深度]
        features = nets[layer].eval(feed_dict={image: input_image_pre})
        if i in [1, 6, 11, 20, 29]:
            convLeft.append(features[0])
        if 1:
            plt.figure(i + 1, figsize=(10, 5))
            plt.matshow(features[0, :, :, 0], cmap=plt.cm.gray, fignum=i + 1)
            plt.title("" + layer)
            plt.axis('off')
            # plt.colorbar()
            plt.show()

# with tf.Session() as sess1:
#     imageB = tf.placeholder('float', shape=shapeB)
#     nets, mean_pixel, all_layers = net(VGG_PATH, imageB)
#     input_image_pre = np.array([preprocess(input_imageB, mean_pixel)])
#     layers = all_layers # For all layers 
#     for i, layer in enumerate(layers):
#         #feature:[batch数，H ,W ,深度]
#         features = nets[layer].eval(feed_dict={imageB: input_image_pre})
#         if i in [1, 6, 11, 20, 29]:
#             convRight.append(features[0])
#
#     matchtFourPatchCoor = pyramidFive(convLeft[4], convRight[4])
#
#     matchtThreePatchCoor = pyramidLevel(matchtFourPatchCoor, convLeft[3], convRight[3], 4)
#
#     matchtTwoPatchCoor = pyramidLevel(matchtThreePatchCoor, convLeft[2], convRight[2], 3)
#
#     matchtOnePatchCoor = pyramidLevel(matchtTwoPatchCoor, convLeft[1], convRight[1], 2)
#
#     matchPoints = pyramidLevel(matchtOnePatchCoor, convLeft[0], convRight[0], 1)
#
#     imgA, imgB = input_image, input_imageB
#     imgTotal = appendImage(imgA, imgB)
#     print(imgA.shape, imgB.shape, imgTotal.shape)
#     from get_descriptor import *
#
#     plt.title("NBB with " + str(len(matchPoints)) + 'points')
#     plt.imshow(imgTotal)
#     for item in matchPoints:
#         plt.plot(item[1], item[0], 'ro')
#         plt.plot(item[3] + imgA.shape[1], item[2], 'ro')
#         plt.plot([item[1], item[3] + imgA.shape[1]], [item[0], item[2]], 'b')
#     plt.show()
#     plot_ransac_match(matchPoints, imgA.shape[1], imgTotal, 'NBB.txt')
#
#
#
#     # filename = 'NBB.txt'
#     # writeFilename(matchPoints, filename)
#     # plt.title("NBB with " + str(len(matchPoints)) + 'points')
#     # plt.imshow(imgTotal[:, :, 0])
#     # for item in matchPoints:
#     #     plt.plot(item[1], item[0], 'ro')
#     #     plt.plot(item[3]+imgA.shape[1], item[2], 'ro')
#     #     plt.plot([item[1], item[3]+imgA.shape[1]], [item[0], item[2]], 'b')
#     # plt.show()
#
#
#
#
#
#
#
#
#     # flag = 0
#     # num = 0
#     # with open("NBB.txt") as file:
#     #     for line in file:
#     #         line = line.rstrip().split(',')
#     #         for i in range(len(line)):
#     #             line[i] = int(line[i])
#     #
#     #         if flag == 0:
#     #             plt.imshow(imgTotal)
#     #
#     #         plt.plot(line[1], line[0], 'ro')
#     #         plt.plot(line[3] + imgA.shape[1], line[2], 'ro')
#     #         plt.plot([line[1], line[3] + imgA.shape[1]], [line[0], line[2]], 'b')
#     #
#     #         if flag == 10:
#     #             # plt.show()
#     #             flag = 0
#     #             plt.savefig('NRR' + str(num) + '.jpg')
#     #         else:
#     #             flag += 1
#     #         num += 1
