from matplotlib import image as mimg
from matplotlib import pyplot as plt
import numpy as np

img = mimg.imread('leena.jpg')
x = img[:, :, 0]
plt.xlabel('Value')
plt.ylabel('pixels freequency')
plt.imshow(x, cmap='gray')
plt.show()  # image show#
plt.title('Histogram for given image')
plt.xlabel('Value')
plt.ylabel('Pixels Freequency')
plt.hist(x)
plt.show()  # histogram show#
