import cv2
import numpy as np


def get_chain_code_features(img):

    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    dir = [0, 0, 0, 0, 0, 0, 0, 0]

    chain_code = ""

    chain_code_pair = np.zeros((8, 8))

    for i in range(len(contours)):
        chain_code = ""
        for j in range(1, len(contours[i]), 1):
            if (contours[i][j]-contours[i][j-1] == np.array([[-1, -1]])).all():
                dir[0] += 1
                chain_code = chain_code+"0"
            elif (contours[i][j]-contours[i][j-1] == np.array([[0, -1]])).all():
                dir[1] += 1
                chain_code = chain_code+"1"
            elif (contours[i][j]-contours[i][j-1] == np.array([[1, -1]])).all():
                dir[2] += 1
                chain_code = chain_code+"2"
            elif (contours[i][j]-contours[i][j-1] == np.array([[1, 0]])).all():
                dir[3] += 1
                chain_code = chain_code+"3"
            elif (contours[i][j]-contours[i][j-1] == np.array([[1, 1]])).all():
                dir[4] += 1
                chain_code = chain_code+"4"
            elif (contours[i][j]-contours[i][j-1] == np.array([[0, 1]])).all():
                dir[5] += 1
                chain_code = chain_code+"5"
            elif (contours[i][j]-contours[i][j-1] == np.array([[-1, 1]])).all():
                dir[6] += 1
                chain_code = chain_code+"6"
            elif (contours[i][j]-contours[i][j-1] == np.array([[-1, 0]])).all():
                dir[7] += 1
                chain_code = chain_code+"7"

        for k in range(1, len(chain_code), 1):
            chain_code_pair[int(chain_code[k-1])][int(chain_code[k])] += 1

    # normalization
    rangeo = np.max(chain_code_pair)-np.min(chain_code_pair)
    chain_code_pair -= np.min(chain_code_pair)
    chain_code_pair = chain_code_pair/rangeo

    dir = np.array(dir).reshape(1, 8)
    rangeo = np.max(dir)-np.min(dir)
    dir -= np.min(dir)
    dir = dir/rangeo

    feature = np.concatenate(
        (chain_code_pair.flatten().reshape([1, 64]), dir), axis=1)

    return feature
