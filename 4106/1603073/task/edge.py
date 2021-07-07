from PIL import Image, ImageFilter
img = Image.open("dinner.jpg")

img.show()
img1 = img.filter(ImageFilter.FIND_EDGES)
img2 = img.filter(ImageFilter.SHARPEN)

img1.show()
img2.show()
