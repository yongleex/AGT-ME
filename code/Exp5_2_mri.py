#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 5.2: Test the AGT-ME with spinal MRI images
Dataset link: https://www.kaggle.com/dutianze/mri-dataset
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import glob
import numpy as np
import matplotlib
from methods.AGT import adaptive_gamma_transform
from methods.CAB import correct_average_brightness
from methods.BIGC import blind_inverse_gamma_correction

matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib
from nibabel.viewers import OrthoSlicer3D


def read_show():
    example_filename = '../images/medical_image_sets/MRI/Case196.nii'

    img = nib.load(example_filename)
    print(img)
    print(img.header['db_name'])

    width, height, queue = img.dataobj.shape

    OrthoSlicer3D(img.dataobj).show()

    img_arr = img.dataobj[:, :, 5]
    plt.imshow(img_arr, cmap='gray')
    plt.show()


def exp():
    image_dir = r"../images/medical_image_sets/MRI/"
    out_dir = r"./temp_out"
    [os.remove(path) for path in glob.glob(out_dir + "/*")]
    image_names = glob.glob(image_dir + "/*")

    for k, img_path in enumerate(image_names):
        # Step 1. read images
        img_raw = nib.load(img_path).dataobj[200:675, :, 5]
        img_raw = cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

        mask = (img_raw > 100).astype(np.uint8) * 255
        img = (255 * (img_raw - np.min(img_raw[:])) / (np.max(img_raw[:]) - np.min(img_raw[:])))
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
            temp_img = cv2.putText(temp_img, name, (0, 70), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)

            if all_imgs is None:
                all_imgs = temp_img.copy()
            else:
                all_imgs = np.concatenate([all_imgs, 255+np.zeros([all_imgs.shape[0], 11, 3])], axis=1)
                all_imgs = np.concatenate([all_imgs, temp_img], axis=1)
        all_imgs = all_imgs.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, img_path.split(os.sep)[-1][:-4] + ".png"), all_imgs)
    os.system("nautilus " + out_dir)


if __name__ == '__main__':
    exp()
    # read_show() # show the MRI image sequence
