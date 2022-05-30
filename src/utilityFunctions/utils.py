import glob
import cv2 as cv
from tqdm import tqdm


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in tqdm(glob.glob(folder + '/*.jpg')):
        img = cv.imread(filename)
        images.append(img)
        if filename.__contains__('M'):
            labels.append(1)
        else:
            labels.append(0)
    return images, labels
  



