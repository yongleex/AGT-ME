#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 3: Test the gamma estimation efficiency in comparison with BIGC method
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
"""
import os
import cv2
import time
import numpy as np
from methods.AGT import adaptive_gamma_transform
from methods.BIGC import blind_inverse_gamma_correction


# exp main function
def exp():
    # Step.0 Set the image path
    image_dir = r"../images/BSD68"
    file_list = os.listdir(image_dir)[0:10]

    size_list = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    agt_time_list = []
    bigc_time_list = []
    for k, name in enumerate(file_list):
        print(str(k) + ":" + name)
        image = cv2.imread(os.path.join(image_dir, name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        agt_time_list.append([])
        bigc_time_list.append([])
        for siz in size_list:
            img = cv2.resize(image, siz)  # Resize the images

            start_time = time.time()
            _, _ = adaptive_gamma_transform(img)
            end_time = time.time()
            agt_time_list[-1].append(end_time - start_time)

            start_time = time.time()
            _, _ = blind_inverse_gamma_correction(img)
            end_time = time.time()
            bigc_time_list[-1].append(end_time - start_time)

    agt_time_list = np.array(agt_time_list)
    bigc_time_list = np.array(bigc_time_list)
    print("AGT time cost:")
    print("    mean:" + str(np.mean(agt_time_list, axis=0)) + "(s)")
    print("    std: " + str(np.std(agt_time_list, axis=0)) + "(s)")

    print("BIGC time cost:")
    print("    mean:" + str(np.mean(bigc_time_list, axis=0)) + "(s)")
    print("    std: " + str(np.std(bigc_time_list, axis=0)) + "(s)")


if __name__ == '__main__':
    exp()
