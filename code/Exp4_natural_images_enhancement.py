#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 4: Test the AGT-ME on the task of natural image contrast enhancement
    The result images are saved in out_dir.
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import glob
import numpy as np
from methods.AGT import adaptive_gamma_transform
from methods.CAB import correct_average_brightness
from methods.BIGC import blind_inverse_gamma_correction


def exp():
    image_dir = r"../images/natural_image_sets"
    out_dir = r"./temp_out/"
    [os.remove(path) for path in glob.glob(out_dir + "/*")]
    image_names = glob.glob(image_dir + "/*/*")

    gamma_list = []
    image_path_list = []
    for k, path in enumerate(image_names):
        # Step 1. read images
        img = cv2.imread(path)
        if img is None or img.shape[2] == 4:  # something wrong to read an image, or BGRA image, or other files
            print("Warning: path (" + path + ") is not valid, we will skip this path...")
            continue

        # Step 2. conduct gamma estimation and image restoration with different methods or config
        _, img_cab = correct_average_brightness(img)  # CAB
        _, img_cab_agt = adaptive_gamma_transform(img, visual=False)  # AGT
        gamma, img_cab_agt_visual = adaptive_gamma_transform(img, visual=True)  # AGT-ME-VISUAL
        _, img_bigc = blind_inverse_gamma_correction(img, "fast", True)  # BIGC

        # Step 3. save the images
        width = img.shape[0]  # 196 for paper eps file
        # names = ["ORIGINAL", "BIGC", "CAB", "AGT-ME", "AGT-ME-VISUAL"]
        names = ["ORIGINAL", "BIGC", "CAB", "AGT-ME"]
        images = [img, img_bigc, img_cab, img_cab_agt, img_cab_agt_visual]
        all_imgs = None
        for name, image in zip(names, images):
            temp_img = cv2.resize(image, (0, 0), fx=width / image.shape[1], fy=width / image.shape[1])
            # temp_img = cv2.putText(temp_img, name, (0, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
            temp_img = cv2.putText(temp_img, name, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 0, 0), 2)
            if all_imgs is None:
                all_imgs = temp_img.copy()
            else:
                all_imgs = np.concatenate([all_imgs, np.zeros([all_imgs.shape[0], 11, 3])+255], axis=1)
                all_imgs = np.concatenate([all_imgs, temp_img], axis=1)
        all_imgs = all_imgs.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, path.split(os.sep)[-1][:-4] + ".png"), all_imgs)
    os.system("nautilus " + out_dir)


    #     # Step 4. record the gammas
    #     gamma_list.append(gamma)
    #     image_path_list.append(path)
    #     print(str(k) + ": " + path + "(" + str(gamma) + ")")
    # # Step 5. analysis the gamma distribution of all images
    # gamma_list = np.array(gamma_list)
    # image_path_list = np.array(image_path_list)
    #
    # index = np.argsort(gamma_list)
    # print(np.concatenate((gamma_list[index][:, np.newaxis], image_path_list[index][:, np.newaxis]), axis=1))
    #
    # # gamma distribution
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.hist(gamma_list, bins=20, range=[0, 2.0])
    # plt.title("Gamma $\gamma_v^*$ distribution of all natural images")
    # gamma_mean = np.mean(gamma_list)
    # gamma_std = np.std(gamma_list)
    # print("gamma mean:" + str(gamma_mean) + "; gamma std:" + str(gamma_std) + "; gamma max:" + str(
    #     np.max(gamma_list)) + "; gamma min:" + str(np.min(gamma_list)))
    # print("Please open the out_dir for details")
    # plt.figure()
    # plt.plot(gamma_list)
    # plt.xlabel("image numbers")
    # plt.ylabel("gamma")
    # plt.show()


if __name__ == '__main__':
    exp()
