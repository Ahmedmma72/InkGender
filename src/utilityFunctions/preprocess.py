import cv2 as cv


def preprocess(img):

    gray = None
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # using adaptive thresholding for each image not a static threshold for all images
    threshold_image = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv.THRESH_BINARY, 199, 5)

    blur = None    # blur the image to remove the noise
    blur = cv.blur(threshold_image, (3, 3))

    return blur
