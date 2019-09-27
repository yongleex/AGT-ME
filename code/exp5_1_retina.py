#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 5.1: Test the AGT-ME with color images of the retina
Dataset link: http://www.isi.uu.nl/Research/Databases/DRIVE/
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import numpy as np
from methods.AGT import adaptive_gamma_transform
from tools.tools import get_all_files, del_file
import matplotlib.pyplot as plt
import PIL.Image as Image


def exp():
    image_dir = r"../images/medical_image_sets/DRIVE/images"
    mask_dir = r"../images/medical_image_sets/DRIVE/mask"
    out_dir = r"./temp_out"
    del_file(out_dir)  # Note that the old image results will be removed using this command
    image_names = get_all_files(image_dir)

    gamma_list = []
    image_path_list = []
    for k, name in enumerate(image_names):
        # Step 1. read images
        img_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, name[:-4] + ".gif")
        img = cv2.imread(img_path)
        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = Image.open(mask_path)
        mask = np.asarray(mask)

        if img is None or img.shape[2] == 4 or mask is None:  # something wrong to read an image, or BGRA image
            print("Warning: path (" + img_path + ") is not valid, we will skip this path...")
            continue
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

        # Step 2. conduct gamma estimation and image restoration with different methods or config
        # gamma, result1 = adaptive_gamma_transform(img, mask=mask, normalize=True,
        #                                           visual=False)  # Method AGT-ME without visual optimization
        gamma, result2 = adaptive_gamma_transform(img, mask=mask, normalize=False,
                                                  visual=True)  # Method AGT-ME with visual optimization
        # _, result3 = blind_inverse_gamma_correction(img, "fast", True)  # Method BIGC with visual optimization

        # Step 3. save the images result and record the gammas
        gamma_list.append(gamma)
        image_path_list.append(name)
        print(str(k) + ": " + name + "(" + str(gamma) + ")")
        # cv2.imwrite(os.path.join(out_dir, name), np.concatenate((img, result1, result2, result3), axis=1))
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1][:-4] + "_agt_me.png"),
                    result2)
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1][:-4] + ".png"), img)
        # cv2.imwrite(os.path.join(out_dir, name)[:-4] + "_bigc.png", result3)

    # Step 4. simple analysis
    gamma_list = np.array(gamma_list)
    image_path_list = np.array(image_path_list)

    index = np.argsort(gamma_list)
    print(np.concatenate((gamma_list[index][:, np.newaxis], image_path_list[index][:, np.newaxis]), axis=1))

    # gamma distribution
    plt.figure()
    plt.hist(gamma_list, bins=20, range=[0, 2.0])
    plt.title("Gamma $\gamma_v^*$ distribution of retina images")
    gamma_mean = np.mean(gamma_list)
    gamma_std = np.std(gamma_list)
    print("gamma mean:" + str(gamma_mean) + "; gamma std:" + str(gamma_std) + "; gamma max:" + str(
        np.max(gamma_list)) + "; gamma min:" + str(np.min(gamma_list)))
    plt.figure()
    plt.plot(gamma_list)
    plt.xlabel("image numbers")
    plt.ylabel("gamma")
    plt.title("go to dir \".\\temp_out\" for all result images ")

    plt.show()


if __name__ == '__main__':
    exp()
