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
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def get_data(num_channels=3, transform=None):
    master_folder = 'data/PhotosDataset/'

    imgs = []
    labels = []
    for photo in os.listdir(master_folder + 'Alex'):
        img = imread(master_folder + '/Alex/' + photo)
        if transform != None:
            img = transform(img)
        imgs.append(resize(img, (200, 200, num_channels)))
        labels.append('Alex')

    for photo in os.listdir(master_folder + 'Hunter'):
        img = imread(master_folder + '/Hunter/' + photo)
        if transform != None:
            img = transform(img)
        imgs.append(resize(img, (200, 200, num_channels)))
        labels.append('Hunter')

    imgs = np.array(imgs)
    labels = np.array(labels)
    return imgs, labels


def get_datagen(X_train):
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range (degrees, 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=.2,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=.2,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    datagen.fit(X_train)
    return datagen


def model_wo_transfer_learning(X_train, y_train, X_test, y_test, epochs, batch_size):
    model = Sequential()

    # Note that we leave the image size as None to allow multiple image sizes
    model.add(Conv2D(96, (7, 7), strides=2, padding='same',
                     activation='relu', input_shape=(200, 200, X_train.shape[-1]),
                     name='input_layer'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(48, (5, 5), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Conv2D(24, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(24, (3, 3), strides=1, padding='same', activation='relu'))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    # Classification layer
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    callbacks = []
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=15, verbose=1, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0))
    datagen = get_datagen(X_train)

    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    hist = model.fit_generator(datagen.flow(X_train, y_train,
                                            batch_size=batch_size),
                               steps_per_epoch=X_train.shape[0] // batch_size,
                               epochs=epochs,
                               validation_data=(X_test, y_test), verbose=1,
                               callbacks=callbacks)
    return model, hist


def model_w_transfer_learning(model_path, X_train, y_train, X_test, y_test, epochs, batch_size):
    model = load_model(model_path)

    # Allow no weight adjustments for the pretrained layers
    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(128, activation='relu', name='final_hidden'))

    # Classification Layer
    model.add(Dense(2, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    callbacks = []
    callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=15, verbose=1, mode='auto',
                                       min_delta=0.0001, cooldown=0, min_lr=0))

    datagen = get_datagen(X_train)
    # opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    hist = model.fit_generator(datagen.flow(X_train, y_train,
                                            batch_size=batch_size),
                               steps_per_epoch=X_train.shape[0] // batch_size,
                               epochs=epochs,
                               validation_data=(X_test, y_test), verbose=1,
                               callbacks=callbacks)

    return model, hist
