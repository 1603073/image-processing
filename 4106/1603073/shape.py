import numpy as np
import cv2

img = cv2.imread("fog.jpg", 0)

cv2.imshow("image", img)

m, n = img.shape
print(img.shape)
