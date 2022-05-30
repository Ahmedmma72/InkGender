import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# for hinge feature
LEG_LENGTH = 25
N_ANGLE_BINS = 12
BIN_SIZE = 360 // N_ANGLE_BINS


def get_hinge_features(bw_img):

    contours, _ = cv.findContours(
        bw_img, cv.RETR_TREE,
        cv.CHAIN_APPROX_NONE
    )
    contours = sorted(contours, key=cv.contourArea, reverse=True)[1:]

    hist = np.zeros((N_ANGLE_BINS, N_ANGLE_BINS))

    for cnt in contours:
        n_pixels = len(cnt)
        if n_pixels <= LEG_LENGTH:
            continue

        points = np.array([point[0] for point in cnt])
        xs, ys = points[:, 0], points[:, 1]
        point_1s = np.array([cnt[(i + LEG_LENGTH) % n_pixels][0]
                            for i in range(n_pixels)])
        point_2s = np.array([cnt[(i - LEG_LENGTH) % n_pixels][0]
                            for i in range(n_pixels)])
        x1s, y1s = point_1s[:, 0], point_1s[:, 1]
        x2s, y2s = point_2s[:, 0], point_2s[:, 1]

        phi_1s = np.degrees(np.arctan2(y1s - ys, x1s - xs) + np.pi)
        phi_2s = np.degrees(np.arctan2(y2s - ys, x2s - xs) + np.pi)

        indices = np.where(phi_2s > phi_1s)[0]

        for i in indices:
            phi1 = int(phi_1s[i] // BIN_SIZE) % N_ANGLE_BINS
            phi2 = int(phi_2s[i] // BIN_SIZE) % N_ANGLE_BINS
            hist[phi1, phi2] += 1

    normalised_hist = hist / np.sum(hist)
    feature_vector = normalised_hist[np.triu_indices_from(
        normalised_hist, k=1)]

    return feature_vector
