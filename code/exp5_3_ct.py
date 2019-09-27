#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 5.3: Test the AGT-ME with Abdominal CT images
Dataset link: https://www.kaggle.com/kmader/ct-scans-before-and-after
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import pydicom  # this package is to read ct original data
import numpy as np
from tools.tools import get_all_files, del_file


def exp():
    image_dir = r"../images/medical_image_sets/CT"
    out_dir = r"./temp_out"
    del_file(out_dir)
    image_names = get_all_files(image_dir)

    for k, name in enumerate(image_names):
        print("processing " + name + "...")
        # Step 1. read CT images with pydicom package
        img_path = os.path.join(image_dir, name)
        dcm = pydicom.read_file(img_path)
        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img = dcm.image.copy()

        if img is None:  # something wrong to read an image, or BGRA image
            print("Warning: path (" + img_path + ") is not valid, we will skip this path...")
            continue

        # Step 2. Get the mask with a simple threshold strategy and normalize it to 0-1
        mask_threshold = -150
        mask = img > mask_threshold
        img_n = (img - mask_threshold) / (np.max(img[:]) - mask_threshold)
        img_n[img_n < 0] = 0

        # Step 3. Apply AGT-ME
        gamma = -1 / (np.nanmean(np.log(img_n[mask])))
        img_agt = np.power(img_n, gamma / 2.2)  # Frankly speaking, I think using gamma is better than gamma/2.2

        # Step 4. Save the results
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1] + "_agt_me.png"),
                    (img_agt * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1] + ".png"),
                    (img_n * 255).astype(np.uint8))
    print("please check the results in dir:" + out_dir)


if __name__ == '__main__':
    exp()
