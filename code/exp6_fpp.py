#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
from methods.AGT import adaptive_gamma_transform
import matplotlib.pyplot as plt

# Step.0 path setting
image_dir = r"../images/slm image"
out_dir = r"./temp_out"

# Step.0 read images
name = os.listdir(image_dir)[0]
path = os.path.join(image_dir, name)
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read as gray image

# Step.1 Gamma restoration
gamma, result1 = adaptive_gamma_transform(img, visual=False)
gamma2, result2 = adaptive_gamma_transform(img, visual=True)
print(gamma, gamma2)

# Step.2 power spectrum analysis
line_index = 50  # 200
# - fft
y1 = np.fft.fftshift(np.fft.fft(img[line_index, 250:850].astype(np.float) - np.mean(img[line_index, 250:850])))
y2 = np.fft.fftshift(np.fft.fft(result1[line_index, 250:850].astype(np.float) - np.mean(result1[line_index, 250:850])))
y3 = np.fft.fftshift(np.fft.fft(result2[line_index, 250:850].astype(np.float) - np.mean(result2[line_index, 250:850])))

ps1 = np.abs(np.sqrt(y1 * np.conjugate(y1)))
ps2 = np.abs(np.sqrt(y2 * np.conjugate(y2)))
ps3 = np.abs(np.sqrt(y3 * np.conjugate(y3)))

# Step.3 display and save results
# - plot the curves at line 50
plt.figure(figsize=(8, 4))
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.98)
x = np.linspace(0, img.shape[1] - 1, img.shape[1])
legends = ["AGT-ME-VISUAL", "AGT-ME", "origin"]
colors = ["darkgreen", "b", "k"]
for k, image in enumerate([result2, result1, img]):
    # plt.plot(x, image[50, :], "k--", label=None)
    plt.plot(x[250:850], image[line_index, 250:850], "-", color=colors[k], label=legends[k])
plt.xlabel("n", fontsize=16)
plt.ylabel("intensity value", fontsize=16)
plt.tick_params(labelsize=16)
plt.legend(loc="upper right", fontsize=16)
plt.xlim((250, 850))
plt.ylim((0, 255))
foo_fig = plt.gcf()
foo_fig.savefig("Fig10d.eps", format='eps', dpi=1200)

# - display the spectrum
x = np.linspace(-len(y1) / 2, len(y1) / 2 - 1, len(y1))
plt.figure(figsize=(8, 4))
plt.subplots_adjust(left=0.15, right=0.99, bottom=0.15, top=0.98)
plt.plot(x, ps3, "-", color="darkgreen", label="AGT-ME-VISUAL")
plt.plot(x, ps2, "b-", label="AGT-ME")
plt.plot(x, ps1, "k-", label="origin")

plt.xlim([0, 150])
plt.ylim([-1, 7E3])
plt.legend(loc="upper right", fontsize=16)
plt.xlabel("frequency(Hz)", fontsize=16)
plt.ylabel("power/frequency(dB/Hz)", fontsize=16)
plt.tick_params(labelsize=16)
foo_fig = plt.gcf()
foo_fig.savefig("Fig10e.eps", format='eps', dpi=1200)

# - save the images
cv2.imwrite(os.path.join(out_dir, name), np.concatenate((img, result1, result2), axis=1))
for k, image in enumerate([img, result1, result2]):  # add a blue line at line 50
    temp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    temp = cv2.line(temp, (250, line_index), (850, line_index), (255, 0, 0), thickness=3)
    cv2.imwrite(os.path.join(out_dir, str(k) + ".png"), temp)

# - display all the images in figure
blank = np.zeros_like(result1)[:, 0:50] + 255
full_image = np.concatenate((img, blank, result1, blank, result2), axis=1)
plt.figure(figsize=(15, 5))
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
plt.imshow(full_image, cmap="gray")  # images
# gray curves
curves = full_image[line_index, :].astype(np.float)
curves[curves == 255] = np.NaN
curves2 = np.zeros_like(curves)
curves2[:] = np.NaN
curves2[250:850] = curves[250:850]
curves2[1024 + 50 + 250:1024 + 50 + 850] = curves[1024 + 50 + 250:1024 + 50 + 850]
curves2[2048 + 100 + 250:2048 + 100 + 850] = curves[2048 + 100 + 250:2048 + 100 + 850]

x = np.linspace(0, curves.shape[0] - 1, curves.shape[0])
plt.plot(x, -curves + 255 + full_image.shape[0], "k-")
plt.plot(x, -curves2 + 255 + full_image.shape[0], "b-")
plt.plot(x, 0 * curves2 + line_index, "b-")
plt.text(0, 0, "ORIGIN")
plt.text(1024 + 50, 0, "AGT-ME")
plt.text(2048 + 100, 0, "AGT-ME-VISUAL")
plt.yticks([])
plt.xticks([])
plt.xlim([0, full_image.shape[1]])
plt.ylim([full_image.shape[0] + 255, 0])
foo_fig = plt.gcf()
foo_fig.savefig("Fig10abc.eps", format='eps', dpi=1200)
plt.show()
