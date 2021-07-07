from PIL import Image, ImageFilter

img = Image.open('plash.jpg')

img1 = img.resize((img.width, img.height), Image.ANTIALIAS)
img1.save('resize.jpg')
