#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 0: Test the gamma distribution of a dataset.
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.10
"""

import os
import cv2
import numpy as np
from methods.AGT import adaptive_gamma_transform
# from methods.BIGC import blind_inverse_gamma_correction
import matplotlib.pyplot as plt

image_dir = r"../images/natural_image_sets/CBSD68/"

out_dir = r"./temp_out"
image_names = os.listdir(image_dir)
gamma_list = []
for k, name in enumerate(image_names):
    path = os.path.join(image_dir, name)
    # img = cv2.imread(path)  # read as gray image
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)  # for non utf-8 image path

    # get the gamma^*
    gamma, result1 = adaptive_gamma_transform(img, visual=False)
    print(gamma, np.mean(img[:]))
    gamma_list.append(gamma)
    print(str(k) + ": " + name + "(" + str(gamma) + ")")

    # save the images to the out_dir
    _, result2 = adaptive_gamma_transform(img, visual=True)
    cv2.imwrite(os.path.join(out_dir, name), np.concatenate((img, result1, result2), axis=1))

gamma_list = np.array(gamma_list)
gamma_mean = np.mean(gamma_list)
gamma_std = np.std(gamma_list)
print("gamma mean:" + str(gamma_mean) + "; gamma std:" + str(gamma_std))
plt.figure()
plt.plot(gamma_list)
plt.xlabel("image numbers")
plt.ylabel("gamma")
plt.show()
