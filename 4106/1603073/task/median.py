from PIL import Image, ImageFilter

img = Image.open("dinner.jpg")

img1 = img.filter(ImageFilter.MedianFilter(size=5))

img1.show()
