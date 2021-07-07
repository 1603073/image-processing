# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:33:39 2021

@author: anan_03066
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def entropy(D):
    marg = np.histogramdd(np.ravel(D), bins=256)[0]/D.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    ntrpy = -np.sum(np.multiply(marg, np.log2(marg)))
    return ntrpy


def sptl2(D, c):
    const = 10
    [row, col] = np.shape(D)
    pred1 = np.zeros((row+1, col+1))
    pred11 = np.zeros((row+1, col+1))

    pred1[0, :] = const*np.ones((1, col+1))
    pred1[1:row+1, 0] = const

    pred1[1:row+1, 1:col+1] = D

    # locoi
    if c == 0:

        for i in range(1, row):
            for j in range(1, col):
                if pred1[i-1][j-1] >= max(pred1[i][j-1], pred1[i-1][j]):
                    pred11[i][j] = min(pred1[i][j-1], pred1[i-1][j])
                elif pred1[i-1][j-1] <= min(pred1[i][j-1], pred1[i-1][j]):
                    pred11[i][j] = max(pred1[i][j-1], pred1[i-1][j])
                else:
                    pred11[i][j] = pred1[i][j-1] + \
                        pred1[i-1][j] - pred1[i-1][j-1]
        pred11 = pred11[1:row, 1:col]
        pred11 = pred11.astype('int64')
        pred11 = pred11.astype('double')
        print("LOCO_I = ", end="")
        return pred11
    # (A+B)/2
    elif c == 1:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i, j] = 0.5*(pred1[i, j-1]+pred1[i-1, j])
        print("(A+B)/2 = ", end="")
    # B
    elif c == 2:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j]
        print("B = ", end="")
    # C
    elif c == 2:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j-1]
        print("C = ", end="")
    # A
    elif c == 3:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1]
        print("A = ", end="")
    # A+B-C
    elif c == 4:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1] + pred1[i-1][j] - pred1[i-1][j-1]
        print("A+B-C = ", end="")
    # A+((B-C)/2)
    elif c == 5:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i][j-1] + \
                    (0.5 * (pred1[i-1][j] - pred1[i-1][j-1]))
        print("A+((B-C)/2) = ", end="")
    # B+((A-C)/2)
    else:
        for i in range(1, row):
            for j in range(1, col):
                pred11[i][j] = pred1[i-1][j] + \
                    (0.5 * (pred1[i][j-1] - pred1[i-1][j-1]))
        print("B+((A-C)/2) = ", end="")

    pred11 = pred11[1:row+1, 1:col+1]
    pred11 = D-np.fix(pred11)
    return pred11


img = mpimg.imread('boat.jpg')
img = img[:, :, 0]
print("original = ", entropy(img), "\n")

for i in range(7):
    sptl = sptl2(img, i)
    print(entropy(sptl), "\n")
