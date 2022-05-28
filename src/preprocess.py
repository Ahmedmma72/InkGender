import cv2 as cv
import numpy as np
from PIL import ImageEnhance


def preprocess_image(img, sharpnessFactor=10, bordersize=3):
    enhancer = ImageEnhance.Sharpness(img)
    enhancedImg = enhancer.enhance(sharpnessFactor)
    (width, height) = (img.width * 2, img.height * 2)
    enhancedImg = enhancedImg.resize((width, height))
    image = np.array(enhancedImg)
    image = cv.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv.BORDER_CONSTANT,
        value=[255, 255, 255]
    )
    orig_image = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.GaussianBlur(image, (3, 3), 0)
    (thresh, bw_image) = cv.threshold(image, 128,
                                      255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    return bw_image, orig_image
