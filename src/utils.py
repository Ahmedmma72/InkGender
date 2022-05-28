import glob
from PIL import Image, ImageEnhance
import cv2 as cv


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in glob.glob(folder + '/*.jpg'):
        img = Image.open(filename)
        images.append(img)
        if filename.__contains__('M'):
            labels.append(1)
        else:
            labels.append(0)
    return images, labels



