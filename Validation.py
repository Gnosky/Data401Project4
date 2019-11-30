import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, MaxPooling2D)
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold

from AlexHunterNet import *


def canny_transform(img):
    return cv2.Canny(img, 200, 200)


K = 2
num_channels = 1
batch_size = 4
epochs = 10
model_path = "models/caltech_101_canny_no_pyramid.h5"

imgs, labels = get_data(num_channels, canny_transform)
encoded_labels = np.array(
    [[1, 0] if label == 'Hunter' else [0, 1] for label in labels])

y_tests = []
reg_preds = []
trans_preds = []

skf = StratifiedKFold(n_splits=K)
# train_index, test_index = next(skf.split(imgs, labels))
for train_index, test_index in skf.split(imgs, labels):
    X_train, X_test = imgs[train_index], imgs[test_index]
    X_train = np.stack(X_train, axis=0)
    X_test = np.stack(X_test, axis=0)
    y_train, y_test = encoded_labels[train_index], encoded_labels[test_index]

    reg_model, reg_hist = model_wo_transfer_learning(
        X_train, y_train, X_test, y_test, epochs, batch_size)

    trans_model, trans_hist = model_w_transfer_learning(
        model_path, X_train, y_train, X_test, y_test, epochs, batch_size)

    reg_pred = []
    trans_pred = []
    for img in X_test:
        trans_pred.append(trans_model.predict(np.expand_dims(img, axis=0)))
        reg_pred.append(reg_model.predict(np.expand_dims(img, axis=0)))

    y_test = ["Hunter" if x[0] > x[1] else "Alex" for x in y_test]
    reg_pred = ["Hunter" if x[0][0] > x[0][1] else "Alex" for x in reg_pred]
    trans_pred = ["Hunter" if x[0][0] > x[0]
                  [1] else "Alex" for x in trans_pred]

    y_tests += y_test
    reg_preds += reg_pred
    trans_preds += trans_pred

print("Model with No transfer learning")
print(classification_report(y_tests, reg_preds))
print("Model with transfer learning")
print(classification_report(y_tests, trans_preds))
