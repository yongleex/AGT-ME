#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Experiment 2: Test the gamma estimation accuracy in comparison with BIGC method
Author: Yong Lee
E-Mail: yongli.cv@gmail.com
C-Data: 2019.04.11
______________________________
version 2
M-Data: 2020.09.03
    1. Correct the bugs of AGT-ME
    2. Add CAB algorithm
______________________________
version 3
M-Data: 2020.11.15
    1. Treat the HE as distortion-free image, additional exp for reviewer response.
"""
import os
import cv2
import numpy as np
from methods.AGT import adaptive_gamma_transform
from methods.BIGC import blind_inverse_gamma_correction
from methods.CAB import correct_average_brightness
import matplotlib.pyplot as plt

MAX_NUMBER = -1  # -1: all image employed; 2: accelerate for you to test it


# support function, add gamma distortion to an image
def gamma_trans(image, gamma):
    """
    Add gamma to an image
    """
    # Step 0. Check the inputs
    # -image
    if np.ndim(image) == 3 and image.shape[-1] == 3:  # color image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        img = hsv[:, :, 2]  # Add gamma distortion on V-channel of HSV colorspace
        color = True
    elif np.ndim(image) == 2:  # gray image
        img = image
        color = False
    else:
        print("ERROR：check the input image of AGT function...")
        return None

    # Step 1. Normalised the image to (0,1), and apply gamma transformation
    img = (img + 0.5) / 256
    img = np.power(img, gamma)
    img = np.clip(img * 256 - 0.5, 0, 255).round().astype(np.uint8)
    if color:
        hsv[:, :, 2] = img
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def linear_calibration(img):
    equ1 = cv2.equalizeHist(img)
    equ2 = 255 - cv2.equalizeHist(255 - img)
    equ = (0.0 + equ1 + equ2) / 2 + np.random.uniform(-0.1, 0.1, equ1.shape)
    equ = np.round(equ).astype(np.uint8)
    return equ


# exp main function
def exp():
    # Step.0 Set the image dataset dir and a dir to store the results
    image_dir = r"../images/BSD68/"
    image_names = os.listdir(image_dir)[:MAX_NUMBER]  # Get the file name of the test images

    out_dir = r"./temp_out"

    # Step.1 the gamma set definition
    gamma_set = np.linspace(0.1, 3.0, 30)
    for gamma in gamma_set:  # save the distorted image for fun
        path = os.path.join(image_dir, image_names[0])
        img = cv2.imread(path, -1)
        distorted_img = gamma_trans(img, gamma)
        cv2.imwrite(out_dir + os.sep + str(gamma) + ".png", distorted_img)

    # Step.2 get the estimated gamma with different methods
    agt_gamma_estimated_list = []
    cab_gamma_estimated_list = []
    bigc_gamma_estimated_list = []
    for gamma in gamma_set:
        agt_gamma_estimated_list.append([])
        cab_gamma_estimated_list.append([])
        bigc_gamma_estimated_list.append([])
        for k, name in enumerate(image_names):
            print("gamma:" + str(gamma) + "; number:" + str(k) + ": " + name)
            path = os.path.join(image_dir, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read as gray image
            # img = cv2.equalizeHist(img)
            img = linear_calibration(img)
            distorted_img = gamma_trans(img, gamma)  # Add gamma distortion
            gamma_origin = 1.0

            # Test with AGT method
            gamma_estimated, _ = adaptive_gamma_transform(distorted_img, visual=False)  # distorted gamma value
            agt_gamma_estimated_list[-1].append(gamma_origin / gamma_estimated)

            # Test with CAB method
            gamma_estimated, _ = correct_average_brightness(distorted_img)  # distorted gamma value
            cab_gamma_estimated_list[-1].append(gamma_origin / gamma_estimated)

            # Test with BIGC method
            gamma_estimated, _ = blind_inverse_gamma_correction(distorted_img, visual=False)  # distorted gamma value
            bigc_gamma_estimated_list[-1].append(gamma_origin / gamma_estimated)

    agt_gamma_estimated_list = np.array(agt_gamma_estimated_list)
    cab_gamma_estimated_list = np.array(cab_gamma_estimated_list)
    bigc_gamma_estimated_list = np.array(bigc_gamma_estimated_list)

    # figure 1. actual gamma VS estimated gamma (AGT method)
    for k, gamma_list in enumerate([agt_gamma_estimated_list, cab_gamma_estimated_list, bigc_gamma_estimated_list]):
        plt.figure()
        plt.subplots_adjust(bottom=0.14, left=0.14)
        plt.grid()
        gamma_set_temp = np.repeat(gamma_set[:, np.newaxis], gamma_list.shape[1], axis=1)
        plt.scatter(gamma_set_temp, gamma_list, s=20, c='b')
        plt.xlim([-0.2, 3.2])
        plt.ylim([-0.2, 3.2])
        plt.xlabel("Gamma truth $\gamma_{truth}$", fontsize=16)
        plt.ylabel("Recognized gamma $\gamma_r$", fontsize=16)
        plt.tick_params(labelsize=16)
        # plt.legend(fontsize=20)
        foo_fig = plt.gcf()
        foo_fig.savefig(f"figs/Fig4a{k}.eps", format='eps', dpi=1200)
        plt.title("set-estimate")

    # figure 2. RMSE error curve for different methods
    gamma_set = np.repeat(gamma_set[:, np.newaxis], gamma_list.shape[1], axis=1)
    rmse_agt = np.sqrt(np.mean((agt_gamma_estimated_list - gamma_set) ** 2, axis=1))
    rmse_cab = np.sqrt(np.mean((cab_gamma_estimated_list - gamma_set) ** 2, axis=1))
    temp = (bigc_gamma_estimated_list - gamma_set) ** 2
    print("Outlier percentage of BIGC:", 100.0 * np.sum(temp[:] > 0.25) / temp.size, "%")
    temp[temp > 0.5] = np.NaN  # special treatment for BIGC outliers
    rmse_bigc = np.sqrt(np.nanmean(temp, axis=1))
    rmse_agt[rmse_agt < 1e-3] = 1e-3
    rmse_cab[rmse_cab < 1e-3] = 1e-3
    rmse_bigc[rmse_bigc < 1e-3] = 1e-3
    print("mean rmse:", np.mean(rmse_agt[:]), "(AGT-ME) and ", np.mean(rmse_cab[:]), "(CAB) and ",
          np.mean(rmse_bigc[:]), "(BIGC)")

    # print(rmse_agt, rmse_bigc)
    plt.figure()
    plt.grid()
    plt.subplots_adjust(bottom=0.14, left=0.18)
    plt.plot(gamma_set[:, 0], rmse_bigc, "r-.", label="BIGC")
    plt.plot(gamma_set[:, 0], rmse_cab, "r:", label="CAB")
    plt.plot(gamma_set[:, 0], rmse_agt, "b-", label="AGT-ME")
    plt.xlim([0.0, 3.0])
    plt.ylim([1e-3, 1.0])
    plt.semilogy()
    plt.xlabel("Gamma truth $\gamma_{truth}$", fontsize=16)
    plt.ylabel("RMSE", fontsize=16)
    plt.tick_params(labelsize=16)
    plt.legend(fontsize=16)

    foo_fig = plt.gcf()
    foo_fig.savefig("figs/Fig4b.eps", format='eps', dpi=1200)

    plt.show()


if __name__ == '__main__':
    exp()

# Result:
# gamma:3.0; number:65: test066.png
# 3.4586181640625
# 8.997802734375
# gamma:3.0; number:66: test067.png
# 0.8321435546874999
# 0.2752783203125
# Outlier percentage of BIGC: 26.915422885572138 %
# mean rmse: 0.04994674455985409 (AGT-ME) and  0.15290822550886457 (BIGC)
