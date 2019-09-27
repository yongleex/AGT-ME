#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
This is the implementation of our method AGT-ME.
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.10
"""
import cv2
import time
import numpy as np


def adaptive_gamma_transform(image, guidance=None, mask=None, normalize=False, visual=True):
    """
    :param image:  input image, color (3 channels) or gray (1 channel);
    :param guidance:  using guidance image to calc gamma value, default is the input image;
    :param mask:  calc gamma value in the mask area, default is the whole image;
    :param normalize: normalize the input with max/min or not
    :param visual: for better visualization, we divide the  maximized entropy gamma with a constant 2.2
    :return: gamma, and output
    """

    # Step 1. Check the inputs: image
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]
        color_flag = True
    elif np.ndim(image) == 2:  # gray image
        img = image
        color_flag = False
    else:
        print("ERROR：check the input image of AGT function...")
        return 1, None
    if not (visual or not visual):
        print("ERROR：check the visual of AGT function...")
        return 1, None

    # Step 2. pre-processing
    if normalize:  # normalization
        img = img.astype(np.float)
        img = (255 * (img - np.min(img[:])) / (np.max(img[:]) - np.min(img[:]) + 0.1)).astype(np.float)
    if guidance is None:  # guidance
        guidance = img
        pass
    elif np.ndim(guidance) == 3 and guidance.shape[-1] == 3:
        guidance = cv2.cvtColor(guidance, cv2.COLOR_BGR2GRAY)
    elif np.ndim(guidance) == 2:  # gray image
        guidance = guidance
    else:
        print("ERROR：check the input guidance of AGT function...")
        return 1, None
    if np.any(guidance.shape != img.shape):
        print("ERROR：guidance size" + guidance.shape + " and image size" + img.shape + " not equal.")
        return 1, None

    # Step 3. Main steps of AGT-ME
    # Step 3.1 image normalization to range (0,1)
    img = (img + 0.5) / 256
    if guidance is None:
        guidance = img
    else:
        guidance = (guidance + 0.5) / 256

    # Step 3.2 calculate the gamma
    ln_guidance = np.log(guidance)
    if mask is not None:
        mask[mask < 255] = 0
        ln_guidance[mask == 0] = np.NaN
    gamma = -1 / np.nanmean(ln_guidance[:])

    # Step 3.3  weather optimize for human visual system
    if visual:
        gamma = gamma / 2.2

    # Step 3.4 apply gamma transformation
    output = np.power(img, gamma)

    # Step 4.0 stretch back and post-process
    output = (output * 256 - 0.5).astype(np.uint8)
    if color_flag:
        hsv[:, :, 2] = output
        output = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return gamma, output


# test function
def simple_example():
    image = cv2.imread(r"../../images/natural_image_sets/CBSD68/14037.png")
    guidance = None  # image with the same size
    visual = True

    start_time = time.time()
    gamma, output = adaptive_gamma_transform(image, guidance=guidance, visual=visual)
    end_time = time.time()

    print("Estimated gamma =" + str(gamma) + ", with time cost=" + str(end_time - start_time) + "s")
    cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("input", image)
    cv2.imshow("output", output)
    cv2.waitKey()


if __name__ == '__main__':
    simple_example()
