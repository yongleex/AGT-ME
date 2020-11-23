#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from methods.AGT import adaptive_gamma_transform
from methods.CAB import correct_average_brightness
from methods.BIGC import blind_inverse_gamma_correction


def spectrum(image):
    y = np.fft.fftshift(np.fft.fft(image[line_index, 250:850].astype(np.float) - np.mean(image[line_index, 250:850])))
    spec = np.abs(np.sqrt(y * np.conjugate(y)))
    return spec


def line_value(image):
    return image[line_index, 250:850].astype(np.float)


# Step.0 path setting
image_dir = r"../images/slm image"
out_dir = r"./temp_out"
[os.remove(path) for path in glob.glob(out_dir + "/*")]

# Step.1 read images
path = glob.glob(image_dir + "/*")[0]
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read as gray image

# Step 2. conduct gamma estimation and image restoration with different methods or config
_, img_cab = correct_average_brightness(img)  # CAB
_, img_agt = adaptive_gamma_transform(img, visual=False)  # AGT
gamma, img_agt_visual = adaptive_gamma_transform(img, visual=True)  # AGT-ME-VISUAL
_, img_bigc = blind_inverse_gamma_correction(img, "fast", True)  # BIGC

# Step.2 power spectrum analysis
names = ["ORIGINAL", "BIGC", "AGT-ME", "CAB"]
images = [img, img_bigc, img_agt, img_cab]
styles = ["g-", "r-.", "b-", "r:"]

line_index = 50  # 200
power_spec = [spectrum(temp) for temp in images]
line_values = [line_value(temp) for temp in images]

# Step.3 display the curves (intensity curve and power spectrum)
plt.figure(figsize=(6, 3))
for line, style, label in zip(line_values, styles, names):
    plt.plot(np.arange(250, 850), line, style, label=label)
plt.xlim(250, 850)
plt.xlabel("n")
plt.ylabel("Intensity value")
plt.legend(loc='upper right')
plt.grid()
foo_fig = plt.gcf()
foo_fig.savefig("figs/Fig8e.eps", format='eps', dpi=1200)

plt.figure(figsize=(6, 3))
styles = ["g-", "r-.", "b-", "r:"]
for ps, style, label in zip(power_spec, styles, names):
    x = np.linspace(-len(ps) / 2, len(ps) / 2 - 1, len(ps))
    plt.plot(x, ps, style, label=label)
plt.xlim([0, 150])
plt.ylim([-1, 7E3])
plt.xlabel("frequency (Hz)")
plt.ylabel("power/frequency (dB/Hz)")
plt.legend(loc='upper right')
foo_fig = plt.gcf()
foo_fig.savefig("figs/Fig8f.eps", format='eps', dpi=1200)
plt.show()

# Step.4 save the images
for t_img, label in zip(images, names):
    temp = cv2.cvtColor(t_img, cv2.COLOR_GRAY2BGR)
    temp = cv2.line(temp, (250, line_index), (850, line_index), (255, 0, 0), thickness=3)
    temp_img = cv2.putText(temp, label, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0), 1)
    cv2.imwrite(os.path.join(out_dir, label + ".png"), temp_img)

os.system("nautilus " + out_dir)
