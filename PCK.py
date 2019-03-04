import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io
from PIL import Image
import os
import xlrd

PrePath = 'D:/Code_Image_Stitch/Vgg19Net/test_data/'

alpha = 0.1
alpha1 = 0.05

def singleToThree(a):
    image = np.expand_dims(a, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    return image

def load_image(path):
    # load image
    img = skimage.io.imread(path)
    flagImg = Image.fromarray(np.uint8(img))
    flagImg = flagImg.resize((224, 224))
    flagImg = np.asarray(flagImg)
    resized_img = flagImg / 255.0

    if len(resized_img.shape) == 2:
        resized_img = singleToThree(resized_img)
        img = singleToThree(img)  # transfer single Channel to three channels

    return resized_img, img

def onPress(event):
    print(event.xdata, event.ydata)  # (col, row)
    # plt.plot(event.xdata, event.ydata, '.')
    # plt.show()
    filename = 'writeCoorLeft.txt'
    with open(filename, 'a') as f:  # the first img
        f.write(str(event.ydata) + '\n')
        f.write(str(event.xdata) + '\n')

def on_press(event):
    print(event.xdata, event.ydata)
    # plt.plot(event.xdata, event.ydata, '.')
    filename = 'writeCoorRight.txt'
    with open(filename, 'a') as f:  # the second img
        f.write(str(event.ydata) + '\n')
        f.write(str(event.xdata) + '\n')


def handMake(img, filename, flag):
    fig = plt.figure()
    plt.imshow(img, animated=True)

    if flag == 0:
        fig.canvas.mpl_connect('button_press_event', onPress)  # first img
    else:
        fig.canvas.mpl_connect('button_press_event', on_press)  # second img

    plt.show()

    coordinates = []
    with open(filename) as f:
        for line in f:  # read the coordinate file line by line
            # print(line.rstrip())
            coordinates.append(float(line.rstrip()))  # add the information into list by the sequence of col after row

    i = 0
    ans = np.ones((3, len(coordinates)//2))
    while i < len(coordinates):
        ans[0][i//2], ans[1][i//2] = int(coordinates[i+1]), int(coordinates[i])
        i += 2

    os.remove(filename)  # delete the coordinate file
    return ans  # return the handMake coordinates

def Jmax(a, b):
    return a if a > b else b

def pck(img1, img2, coordinates_left, coordinates_right, Hom):

    ################################add the handMake coordinates###########################
    # coordinates_left = handMake(img1, 'writeCoorLeft.txt', 0)  # 3 * num (col, row, 1)
    # coordinates_right = handMake(img2, 'writeCoorRight.txt', 1)
    ################################add the handMake coordinates###########################

    assert len(coordinates_left) == len(coordinates_right)

    coordinates_right_After = np.dot(Hom, coordinates_right)

    print(coordinates_right.shape, coordinates_left.shape)
    coordinates_right_After[0, :], coordinates_right_After[1, :] = coordinates_right_After[0,:] / coordinates_right_After[2,:], coordinates_right_After[1,:] / coordinates_right_After[2,:]

    error = []
    lac = 0
    for i in range(coordinates_left.shape[1]):
        error.append(np.sqrt((coordinates_right_After[0][i] - coordinates_left[0][i]) ** 2 + (
                coordinates_right_After[1][i] - coordinates_left[1][i]) ** 2))
        lac += abs((coordinates_right_After[0][i] - coordinates_left[0][i]) / img1.shape[1]) + abs(\
            coordinates_right_After[1][i] - coordinates_left[1][i]) / img1.shape[0]
    num = 0
    num1 = 0
    for err in error:
        if err < (alpha * Jmax(img1.shape[0], img1.shape[1])):
            num += 1
        if err < (alpha1 * Jmax(img1.shape[0], img1.shape[1])):
            num1 += 1
    return num / coordinates_left.shape[1], num1 / coordinates_left.shape[1], lac