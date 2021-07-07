from PIL import Image
import cv2

img = cv2.imread("../boat.jpg", 0)
img_size = img.size
print(f"Image file size: {img_size} bits")
print(img.shape[0])
