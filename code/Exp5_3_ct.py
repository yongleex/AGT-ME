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
import glob
import pydicom
import numpy as np

from methods.AGT import adaptive_gamma_transform
from methods.CAB import correct_average_brightness
from methods.BIGC import blind_inverse_gamma_correction


def exp():
    image_dir = r"../images/medical_image_sets/CT"
    out_dir = r"./temp_out"
    [os.remove(path) for path in glob.glob(out_dir + "/*")]
    image_names = glob.glob(image_dir + "/*")

    for k, img_path in enumerate(image_names):
        print("processing " + img_path + "...")
        # Step 1. read CT images with pydicom package
        dcm = pydicom.read_file(img_path)
        dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img_raw = dcm.image.copy()

        if img_raw is None:  # something wrong to read an image, or BGRA image
            print("Warning: path (" + img_path + ") is not valid, we will skip this path...")
            continue

        # Step 2. Get the mask with a simple threshold strategy and normalize it to 0-1
        mask_threshold = -150
        mask = ((img_raw > mask_threshold) * 255).astype(np.uint8)
        img = 255 * (img_raw - mask_threshold) / (np.max(img_raw[:]) - mask_threshold)
        img[img < 0] = 0
        img = img.round().astype(np.uint8)

        # Step 2. conduct gamma estimation and image restoration with different methods or config
        _, img_cab = correct_average_brightness(img, mask=mask)  # CAB
        _, img_cab_agt = adaptive_gamma_transform(img, mask=mask, visual=False)  # AGT
        gamma, img_cab_agt_visual = adaptive_gamma_transform(img, mask=mask, visual=True)  # AGT-ME-VISUAL
        _, img_bigc = blind_inverse_gamma_correction(img, "fast", True)  # BIGC
        img_bigc[mask < 255] = 0

        # Step 3. save the images
        width = img.shape[0]  # 196 for paper eps file
        # names = ["ORIGINAL", "BIGC", "CAB", "AGT-ME", "AGT-ME-VISUAL"]
        names = ["ORIGINAL", "BIGC", "CAB", "AGT-ME"]
        images = [img, img_bigc, img_cab, img_cab_agt, img_cab_agt_visual]
        all_imgs = None
        for name, image in zip(names, images):
            temp_img = cv2.resize(image, (0, 0), fx=width / image.shape[1], fy=width / image.shape[1])
            temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2BGR)
            # temp_img = cv2.putText(temp_img, name, (0, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
            temp_img = cv2.putText(temp_img, name, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 2)

            if all_imgs is None:
                all_imgs = temp_img.copy()
            else:
                all_imgs = np.concatenate([all_imgs, 255 + np.zeros([all_imgs.shape[0], 11, 3])], axis=1)
                all_imgs = np.concatenate([all_imgs, temp_img], axis=1)
        all_imgs = all_imgs.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, img_path.split(os.sep)[-1] + ".png"), all_imgs)
        # cv2.imwrite(os.path.join(out_dir, img_path.split(os.sep)[-1] + "mask.png"), mask)
    os.system("nautilus " + out_dir)

    print("please check the results in dir:" + out_dir)
    os.system("nautilus " + out_dir)


if __name__ == '__main__':
    exp()
