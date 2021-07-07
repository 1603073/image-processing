from PIL import Image
from PIL import ImageFilter


def redBand(intensity):
    return 0


def greenBand(intensity):
    return 0


def blueBand(intensity):
    return intensity


img = Image.open("dinner.jpg")
img.show()
img_split = img.split()

redPixel = img_split[0].point(redBand)
greenPixel = img_split[0].point(greenBand)
bluePixel = img_split[0].point(blueBand)

redPixel.show()
greenPixel.show()
bluePixel.show()

new_img = Image.merge("RGB", (redPixel, greenPixel, bluePixel))
new_img.show()
