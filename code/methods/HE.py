#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
This is our implementation of Histogram Equalization (HE).
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.10
"""
import cv2
import time
import numpy as np
# import matplotlib.pyplot as plt


def histogram_equalization(image):
    """
    :param image:  input image, color (3 channels) or gray (1 channel);
    :return: gamma, and result
    """
    # Step 0. Check the inputs
    # -image
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
        color = True
    elif np.ndim(image) == 2:  # gray image
        img = image
        color = False
    else:
        print("ERROR：check the input image of HE function...")
        return 1, None
    result = cv2.equalizeHist(img)

    if color:
        hsv[:, :, 2] = result
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


# test function
def simple_example():
    image = cv2.imread(r"../../images/natural_image_sets/CBSD68/14037.png")
    result = histogram_equalization(image)
    # plt.hist(image.ravel(), 256, [0, 256])
    # plt.show()
    cv2.namedWindow("origin", cv2.WINDOW_NORMAL)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("origin", image)
    cv2.imshow("result", result)
    cv2.waitKey()


if __name__ == '__main__':
    simple_example()
