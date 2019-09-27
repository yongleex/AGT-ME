#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 4: Test the AGT-ME on the task of natural image contrast enhancement
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
# import seaborn as sns
import os
import cv2
import numpy as np
from methods.AGT import adaptive_gamma_transform
from methods.BIGC import blind_inverse_gamma_correction
from tools.tools import get_all_files, del_file
import matplotlib.pyplot as plt


def exp():
    image_dir = r"../images/natural_image_sets"
    out_dir = r"./temp_out/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    del_file(out_dir)  # Note that the old image results will be removed using this command
    # image_names = os.listdir(image_dir)
    image_names = get_all_files(image_dir)

    gamma_list = []
    image_path_list = []
    for k, name in enumerate(image_names):
        # Step 1. read images
        path = os.path.join(image_dir, name)
        img = cv2.imread(path)
        if img is None or img.shape[2] == 4:  # something wrong to read an image, or BGRA image
            print("Warning: path (" + path + ") is not valid, we will skip this path...")
            continue

        # Step 2. conduct gamma estimation and image restoration with different methods or config
        # gamma, result1 = adaptive_gamma_transform(img, visual=False)  # Method AGT-ME without visual optimization
        gamma, result2 = adaptive_gamma_transform(img, visual=True)  # Method AGT-ME with visual optimization
        # _, result3 = blind_inverse_gamma_correction(img, "fast", True)  # Method BIGC with visual optimization

        # Step 3. save the images result and record the gammas
        gamma_list.append(gamma)
        image_path_list.append(name)
        print(str(k) + ": " + name + "(" + str(gamma) + ")")
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1][:-4] + "_agt_me.png"),
                    result2)
        cv2.imwrite(os.path.join(out_dir, ("%.4f" % gamma) + "x" + name.split(os.sep)[-1][:-4] + ".png"), img)

    # Step 4. analysis
    gamma_list = np.array(gamma_list)
    image_path_list = np.array(image_path_list)

    index = np.argsort(gamma_list)
    print(np.concatenate((gamma_list[index][:, np.newaxis], image_path_list[index][:, np.newaxis]), axis=1))

    # gamma distribution
    plt.figure()
    plt.hist(gamma_list, bins=20, range=[0, 2.0])
    plt.title("Gamma $\gamma_v^*$ distribution of all natural images")
    gamma_mean = np.mean(gamma_list)
    gamma_std = np.std(gamma_list)
    print("gamma mean:" + str(gamma_mean) + "; gamma std:" + str(gamma_std) + "; gamma max:" + str(
        np.max(gamma_list)) + "; gamma min:" + str(np.min(gamma_list)))
    print("Please open the out_dir for details")
    plt.figure()
    plt.plot(gamma_list)
    plt.xlabel("image numbers")
    plt.ylabel("gamma")
    plt.title("go to dir \".\\temp_out\" for all result images ")
    plt.show()


if __name__ == '__main__':
    exp()
