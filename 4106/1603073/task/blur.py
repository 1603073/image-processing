from PIL import Image, ImageFilter
from matplotlib import pyplot as plt
img = Image.open('dinner.jpg')
img.show()
img1 = img.filter(ImageFilter.BLUR)
plt.imshow(img1)
plt.show()
