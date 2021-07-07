import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('fog.jpg', 0)  # cv2 to read the image
f1 = np.fft.fft2(img)
f1_shift = np.fft.fftshift(f1)
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)  # Calculate spectrum center
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
f1_shift = f1_shift*mask

f_ishift = np.fft.ifftshift(f1_shift)  # inverse fourier transform
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
img_back = (img_back-np.amin(img_back))/(np.amax(img_back)-np.amin(img_back))

plt.figure(figsize=(15, 15))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original image')
plt.show()
plt.subplot(122)
plt.imshow(img_back, cmap='gray')
plt.title('output image')
plt.show()
