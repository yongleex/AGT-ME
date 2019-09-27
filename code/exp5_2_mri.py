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
import numpy as np
from tools.tools import get_all_files, del_file
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pylab as plt
import nibabel as nib # this package to read the mri original data
# from nibabel import nifti1
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
    del_file(out_dir)  # Note that the old image results will be removed using this command
    image_names = os.listdir(image_dir)

    gamma_list = []
    image_path_list = []
    for k, name in enumerate(image_names):
        # Step 1. read images
        img_path = os.path.join(image_dir, name)
        img = nib.load(img_path).dataobj[200:675, :, 5]
        mask = img > 100

        img_n = (img - np.min(img[:])) / (np.max(img[:]) - np.min(img[:]))

        # Step 3. Apply AGT-ME
        gamma = -1 / (np.nanmean(np.log(img_n[mask])))
        gamma = gamma / 2.2  # Frankly speaking, I think using gamma is better than gamma/2.2
        img_agt = np.power(img_n, gamma)

        print(str(k) + ": " + name + "(" + str(gamma) + ")")
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1][:-4] + "_agt_me.png"),
                    (img_agt * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1][:-4] + ".png"),
                    (img_n * 255).astype(np.uint8))


if __name__ == '__main__':
    exp()
    read_show()
