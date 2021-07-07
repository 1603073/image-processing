import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def entropy(Data):
    marg = np.histogramdd(np.ravel(Data), bins=256)[0]/Data.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    entrpy = -np.sum(np.multiply(marg, np.log2(marg)))
    return entrpy


def loco_i(Data):
    const = 50
    [row, col] = np.shape(Data)
    pred1 = np.zeros((row+1, col+1))
    pred11 = np.zeros((row+1, col+1))

    pred1[0, :] = const*np.ones((1, col+1))
    pred1[1:row+1, 0] = const

    pred1[1:row+1, 1:col+1] = Data

    for i in range(1, row):
        for j in range(1, col):
            if pred1[i-1][j-1] >= max(pred1[i][j-1], pred1[i-1][j]):
                pred11[i][j] = min(pred1[i][j-1], pred1[i-1][j])
            elif pred1[i-1][j-1] <= min(pred1[i][j-1], pred1[i-1][j]):
                pred11[i][j] = max(pred1[i][j-1], pred1[i-1][j])
            else:
                pred11[i][j] = pred1[i][j-1]+pred1[i-1][j] - pred1[i-1][j-1]
    pred11 = pred11[1:row, 1:col]
    pred11 = pred11.astype('int64')
    pred11 = pred11.astype('double')
    print("loco_i = ", end="")
    return pred11


def sptl2(Data):
    const = 10
    [row, col] = np.shape(D)
    pred1 = np.zeros((row+1, col+1))
    pred11 = np.zeros((row+1, col+1))

    pred1[0, :] = const*np.ones((1, col+1))
    pred1[1:row+1, 0] = const

    pred1[1:row+1, 1:col+1] = D

    for i in range(1, row):
        for j in range(1, col):
            pred11[i][j] = 0.5 * (pred1[i][j-1] + pred1[i-1][j])
    print("(A+B)/2 = ", end="")
    pred11 = pred11[1:row+1, 1:col+1]
    pred11 = Data - np.fix(pred11)
    return pred11


def temporal_sptl2(Data, c):
    const = 5
    [row, col] = np.shape(Data)
    pred1 = np.zeros((row+1, col+1))
    pred11 = np.zeros((row+1, col+1))
    pred1[0, :] = const * np.ones((1, col+1))
    pred1[1: row+1, 0] = const
    pred1[1: row+1, 1:col+1] = Data

    if c == 0:  # X=B
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j]

    if c == 1:  # X=C
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j-1]

    if c == 2:  # X=A
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1]

    if c == 3:  # X=(A+B)/2
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = 0.5 * (pred1[i][j-1] + pred1[i-1][j])

    if c == 4:  # X=A+B-C
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1] + pred1[i-1][j] - pred1[i-1][j-1]

    if c == 5:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1] + \
                    (0.5 * (pred1[i-1][j] - pred1[i-1][j-1]))

    if c == 6:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j] + \
                    (0.5 * (pred1[i][j-1] - pred1[i-1][j-1]))

    pred11 = pred11[1:row+1, 1:col+1]
    pred11 = Data - np.fix(pred11)

    return pred11


img = mpimg.imread('boat.jpg')
img = img[:, :, 0]
print("original = ", entropy(img), "\n")

for i in range(7):
    loco = loco_i(img)
    print(entropy(loco_i))
    sp = sptl2(img)
    print(entropy(sp))
    temp = temporal_sptl2(img, i)
    print(entropy(temp), "\n")
