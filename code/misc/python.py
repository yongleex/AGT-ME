import cv2
import numpy as np

img = cv2.imread("./106024.png", 0)

equ = cv2.equalizeHist(img)
cv2.imwrite("gray.png", img)
cv2.imwrite('res.png', equ)

strip = np.zeros([50, 1]) + np.arange(0, 255, 255 / equ.shape[1])[np.newaxis, :]
equ2 = np.vstack((equ, strip))  # stacking images side-by-side

cv2.imwrite('res2.png', equ2)
pass

#
# def equalization(img):
#     equ1 = cv2.equalizeHist(img)
#     equ2 = 255 - cv2.equalizeHist(255 - img)
#     equ = (0.0 + equ1 + equ2) / 2
#     equ = equ.astype(np.uint8)
#     return equ
#
#
# # second test
# x = np.zeros([50, 1]) + np.arange(0, 26, 1)[np.newaxis, :]
# original = np.zeros([50, 1]) + np.arange(0, 26, 1)[np.newaxis, :]
# original[original > 20] = 20
# x = x.astype(np.uint8)
#
# y = x.copy()
# y[y < 5] = 5
# original = original.astype(np.uint8)
# equ_x1 = cv2.equalizeHist(x)
# equ_y1 = cv2.equalizeHist(y)
# equ_o1 = cv2.equalizeHist(original)
# equ_x = equalization(x)
# equ_y = equalization(y)
# equ_o = equalization(original)
# print(original.astype(np.int) - equ)
# print(np.sum(original - equ))
