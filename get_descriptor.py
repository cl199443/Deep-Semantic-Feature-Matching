from PIL import Image
from numpy import *
import numpy as np
from pylab import *

def concatDescriptor(map, row, col):
    if row == 12 or col == 12 or row ==13 or col == 13:
        print(row, col)
    des = map[0][row][col]
    des = np.concatenate([des, map[0][row][col-1]])  # left
    des = np.concatenate([des, map[0][row][col+1]])  # right
    des = np.concatenate([des, map[0][row-1][col]])  # top
    des = np.concatenate([des, map[0][row+1][col]])  # bottom

    # des = np.concatenate([des, map[0][row-1][col-1]])  # left-top
    # des = np.concatenate([des, map[0][row-1][col+1]])  # left-bottom
    # des = np.concatenate([des, map[0][row+1][col-1]])  # right-top
    # des = np.concatenate([des, map[0][row+1][col+1]])  # right-bottom

    assert ((des.shape[0]) == (384*5))
    return des

def get_descriptor(img, coordinates, wid):
    # img_temp = Image.fromarray(np.uint8(img))
    # image = img_temp.convert("L")
    #imge = rgb2gray(img)
    descriptor = []
    # print("axis lenth is : ", len(axis_y))
    for axis in coordinates:
        patch = img[axis[0] - wid:axis[0] + wid + 1, axis[1] - wid:axis[1] + wid + 1, 0:3].flatten()

        if(len(patch) != 0):
            descriptor.append(patch)
        else:
            print("({},{}) is unvalid".format(axis[0], axis[1]))

    return descriptor

def match(left_descriptor, right_descriptor, thrs):
    distance = -ones((len(left_descriptor), len(right_descriptor)))

    for i in range(len(left_descriptor)):
        d1 = (left_descriptor[i] - mean(left_descriptor[i])) / std(left_descriptor[i])
        for j in range(len(right_descriptor)):
            if( len( left_descriptor[i] ) != len( right_descriptor[j] ) ):
                continue
            d2 = (right_descriptor[j]-mean(right_descriptor[j])) / std(right_descriptor[j])
            ncc = sum(d1 * d2) / (len(left_descriptor[0])-1)

            if ncc > thrs:
                distance[i, j] = ncc  # ncc correlation full points to 1
    ndx = argsort(-distance)
    match_scores = ndx[:, 0]

    print("ndx is : ", ndx[:, 0:2])
    return match_scores


def two_match(des1, des2, threhold):
    match12 = match(des1, des2, threhold)

    match21 = match(des2, des1, threhold)

    ndx_12 = where(match12 >= 0)[0]  # return the axis index

    for item in ndx_12:
        if match21[match12[item]] != item:
            match12[item] = -1

    dict = []
    for i, j in enumerate(match12):
        if j > 0:
            dict.append(np.sum((des1[i]-des2[j])**2))



    return match12

def appendimages(im1, im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = concatenate((im1, zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = concatenate((im2, zeros((rows1-rows2, im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.
    assert (im1.shape[0] != im2.shape[0])
    return concatenate((im1, im2), axis=1)

def getHom(point_des, point_src, npoints):  # (right, left)

    ## H * point_des = point_src
    x_des = point_des[0, :].T  # col
    y_des = point_des[1, :].T  # row
    x_src = point_src[0, :].T
    y_src = point_src[1, :].T

    A = np.zeros((npoints * 2, 8))
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

    flag = np.linalg.inv(np.dot(A.T, A))
    flag1 = np.dot(flag, A.T)
    h = np.dot(flag1, B)

    # print("bh-a:", np.dot(A, h)-B)
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

    return ans

def ransac(ref_B, dst_A, npoints, err_thd):  # (right, left)
    pre_number_less_err = 0
    random_num_points = 8
    right_matrix = []

    for i in range(5000):
        random_col = np.random.randint(0, npoints, random_num_points)
        randomB_dot = ref_B[:, random_col]
        randomA_dot = dst_A[:, random_col]

        matrix = getHom(randomB_dot, randomA_dot, random_num_points)

        ref_B_after = np.dot(matrix, ref_B)
        ref_B_after[0, :] = ref_B_after[0, :] / ref_B_after[2, :]
        ref_B_after[1, :] = ref_B_after[1, :] / ref_B_after[2, :]
        ref_B_after[2, :] = ref_B_after[2, :] / ref_B_after[2, :]

        error = (ref_B_after[0, :]-dst_A[0, :])**2 + (ref_B_after[1, :]-dst_A[1, :])**2

        number_less_err = 0
        # print("{} error is : {} ".format(i, error))
        for i in error:
            if(i < err_thd):
                number_less_err = number_less_err + 1

        if(number_less_err >= npoints * 0.90):
            right_matrix = matrix
            break
        else:
            if(number_less_err > pre_number_less_err):
                pre_number_less_err = number_less_err
                right_matrix = matrix
        # print(pre_number_less_err, number_less_err)
        # print(right_matrix)
    return right_matrix

def plot_ransac_match(matchpoints, cols, im3, errorThold, scale):

    # num: the num of wright pre-matched feature points
    # cols：the col value of stitchedImage
    # match_info:list size is len(left_coordinates),both wrong and wright match included

    num = len(matchpoints)
    matcher_A = np.ones((3, num))
    matcher_B = np.ones((3, num))
    number = 0
    err_hold = 35

    for line in matchpoints:
            matcher_A[0][number] = line[1]  # col-y
            matcher_A[1][number] = line[0]  # row-x
            matcher_B[0][number] = line[3]  # col-y
            matcher_B[1][number] = line[2]  # row-x
            number = number + 1
    # for i, j in enumerate(match_info):
    #     if j > -1:
    #         matcher_A[0][number] = left_coordinates[i][1]
    #         matcher_A[1][number] = left_coordinates[i][0]
    #         matcher_B[0][number] = right_coordinates[j][1]
    #         matcher_B[1][number] = right_coordinates[j][0]
    #         number = number + 1

    # print(matcher_B.shape, matcher_A.shape)


    color = ['b', 'c', 'g', 'k', 'm', 'r', 'w', 'y']
    right_matrix = ransac(matcher_B, matcher_A, num, err_hold)
    print(right_matrix)
    # print(matcher_B.shape,matcher_B)
    matcher_B_after = np.dot(right_matrix, matcher_B)
    # print("matcher after is :", matcher_B_after)
    matcher_B_after[0, :] = matcher_B_after[0, :] / matcher_B_after[2, :]
    matcher_B_after[1, :] = matcher_B_after[1, :] / matcher_B_after[2, :]

    error = (matcher_B_after[0, :] - matcher_A[0, :]) ** 2 + (matcher_B_after[1, :] - matcher_A[1, :]) ** 2


    ###############################################True matching####################################################
    flag = 0
    # plt.imshow(im3)
    # plt.title('FInalMatching')
    # plt.axis('off')

    pointsNum = 0
    upFilter = list()
    for i in range(num):
        # print(error[i])
        if (error[i] < errorThold):
            matcher_A[1][i], matcher_A[0][i] = int(matcher_A[1][i] * scale[0]), int(matcher_A[0][i] * scale[1])
            matcher_B[1][i], matcher_B[0][i] = int(matcher_B[1][i] * scale[2]), int(matcher_B[0][i] * scale[3])
            # plt.plot(matcher_A[0][i], matcher_A[1][i], color[flag % 8] + '.')  # show points
            # plt.plot(cols + matcher_B[0][i], matcher_B[1][i], color[flag % 8] + '.')  # show points
            # plt.plot([matcher_A[0][i], cols + matcher_B[0][i]], [matcher_A[1][i], matcher_B[1][i]], color[flag % 8])  # True matching
            flag += 1
            pointsNum += 1
            upFilter.append([int(matcher_A[1][i]), int(matcher_A[0][i]), int(matcher_B[1][i]), int(matcher_B[0][i])])  # save the ransac filtered matching points
            # [row, col, row, col]
    # plt.show()

    ###############################################True matching####################################################

    ###############################################False matching####################################################
    '''
    flag = 0
    plt.imshow(im3)
    plt.title('FalseMatching')
    plt.axis('off')
    for i in range(num):
        if (error[i] > errorThold):
            plt.plot(matcher_A[0][i], matcher_A[1][i], color[flag % 8] + 'o')  # show points
            plt.plot(cols + matcher_B[0][i], matcher_B[1][i], color[flag % 8] + 'o')  # show points
            plt.plot([matcher_A[0][i], cols + matcher_B[0][i]], [matcher_A[1][i], matcher_B[1][i]], color[flag % 8])  # False matching
            flag += 1
    plt.show()
    '''
    ###############################################False matching####################################################
    # print("upFilter is : ", upFilter)
    return upFilter, len(matchpoints), pointsNum, right_matrix