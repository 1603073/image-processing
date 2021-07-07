
from PIL import Image
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
image_path = 'F:\7th semester\cse 4105\4106\assignments\boat.jpg'
image = cv2.imread(image_path, 0)
# cv2_imshow(image)

image_file_size = (os.path.getsize(f'{image_path}')*8)
print(f'Image file size: {image_file_size} bits')

bits_per_pixel = image_file_size / (image.shape[0]*image.shape[1])
print(f"-> {bits_per_pixel} Bits per pixel")


def ImageEntropy(image):
    marg = np.histogramdd(np.ravel(image), bins=256)[0]/image.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entropy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entropy


def loco_i(Data):
    const = 50
    row = Data.shape[0]
    col = Data.shape[1]
    pred1 = np.zeros((row+1, col+1))
    pred11 = np.zeros((row+1, col+1))

    pred1[0, :] = const*np.ones(1, col+1)
    pred1[1:row+1, 0] = const

    pred1[0, :] = const*np.ones(1, col+1)
    pred1[1:rows+1, 0] = const*np.ones((row, 1))
    pred1[1:rows+1, 1:columns+1] = image

    pred1[:row, :col] = Data
    print(pred1)
    print(pred11)

    for i in range(1, row):
        for j in range(1, col):
            if pred1[i-1][j-1] >= max(pred1[i][j-1], pred1[i-1][j]):
                pred11[i][j] = min(pred1[i][j-1], pred1[i-1][j])
            elif pred1[i-1][j-1] <= min(pred1[i][j-1], pred1[i-1][j]):
                pred11[i][j] = max(pred1[i][j-1], pred1[i-1][j])
            else:
                pred11[i][j] = pred1[i][j-1] + pred1[i-1][j] - pred1[i-1][j-1]

    pred11 = pred11[1:row, 1:col]
    print(pred11)
    pred11 = pred11.astype('int64')
    print(pred11)
    pred11 = pred11.astype('double')
    print(pred11)
    return pred11


def sptl2(Data):
    const = 10
    [row, col] = np.shape(Data)
    pred1 = np.zeros((row+1, col+1))
    pred11 = np.zeros((row+1, col+1))

    pred1[0, :] = const * np.ones((1, col+1))
    pred1[1: row+1, 0] = const
    pred1[1: row+1, 1:col+1] = Data

    #X = (A+B)/2
    for i in range(1, row):
        for j in range(1, col):
            pred11[i][j] = 0.5 * (pred1[i][j-1] + pred1[i-1][j])

    pred11 = pred11[1:row+1, 1:col+1]
    pred11 = Data - np.fix(pred11)

    return pred11


def temporal_sptl2(Data, case):
    const = 5
    [row, col] = np.shape(Data)
    pred1 = np.zeros((row+1, col+1))
    pred11 = np.zeros((row+1, col+1))

    pred1[0, :] = const * np.ones((1, col+1))
    pred1[1: row+1, 0] = const
    pred1[1: row+1, 1:col+1] = Data

    if case == 1:  # X=B
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j]

    if case == 2:  # X=C
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j-1]

    if case == 3:  # X=A
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1]

    if case == 4:  # X=(A+B)/2
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = 0.5 * (pred1[i][j-1] + pred1[i-1][j])

    if case == 5:  # X=A+B-C
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1] + pred1[i-1][j] - pred1[i-1][j-1]

    if case == 6:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1] + \
                    (0.5 * (pred1[i-1][j] - pred1[i-1][j-1]))

    if case == 7:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j] + \
                    (0.5 * (pred1[i][j-1] - pred1[i-1][j-1]))

    pred11 = pred11[1:row+1, 1:col+1]
    pred11 = Data - np.fix(pred11)

    return pred11


original_image = plt.imread('/content/drive/MyDrive/B2DBy.jpg')
original_image = original_image[:, :, 0]
residual_image = sptl2(original_image)

entropy_original_image = entropy(original_image)
entropy_residual_image = entropy(residual_image)

residual_entropy_splt2 = entropy(residual_image)
print(
    f'Origianl Entropy: {entropy_original_image} Residual Entropy splt2 [X=(A+B)/2]: {entropy_residual_image}')
temporal_spl2_output = []
temporal_residual_entropy = []

for i in range(0, 8):
    if i == 0:
        temporal_spl2_output.append(-1)
        continue
    temporal_spl2_output.append(temporal_sptl2(original_image, i))


for i in range(0, 8):
    if i == 0:
        temporal_spl2_output.append(-1)
        continue
    temporal_residual_entropy.append(entropy(temporal_spl2_output[i]))

print('Original Entropy: ', entropy_original_image)


for item in temporal_residual_entropy:
    print(f'Case: {temporal_residual_entropy.index(item)+1}: {item}')

img = loco_i(original_image)

(h, w) = img.shape
data = np.zeros((h, w), dtype=np.double)
data = img.copy()
img = Image.fromarray(data, 'L')
img.save('loco_image.jpg')
