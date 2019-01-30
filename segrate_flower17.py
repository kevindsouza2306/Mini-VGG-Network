__author__ = 'kevin'
from imutils import paths
import os

import cv2


def call():
    imagePath = "C:\\Users\\kevin\\PycharmProjects\\Mini-VGG-Network\\flowers17_dataset"
    image_no = 1
    classLabel = 1
    images = list(paths.list_images(imagePath))
    for (k, img) in enumerate(images):
        #image_no += 1
        image = cv2.imread(img)
        imgpth = os.path.sep.join(
            ["C:\\Users\\kevin\\PycharmProjects\\Mini-VGG-Network\\flowers17_dataset", str(classLabel),
             "{}.jpg".format(k)])
        print(imgpth)
        #cv2.imwrite(image, "C:\\Users\\kevin\\PycharmProjects\\Mini-VGG-Network\\flowers17_dataset\dfgh.jpg")
        if k % 80 == 0:
            classLabel += 1
            #image_no = 0
            os.makedirs(classLabel)


call()
