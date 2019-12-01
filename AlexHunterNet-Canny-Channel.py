
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from scipy.misc import imread, imresize
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

# sys.path.append('./keras-spp/')
# from spp.SpatialPyramidPooling import SpatialPyramidPooling



# #### Load Data

master_folder = 'data/PhotosDataset/'

imgs = []
labels = []
for photo in os.listdir(master_folder + 'Alex'):
    img = imread(master_folder + '/Alex/' + photo)
    edges = cv2.Canny(img, 200, 200)
    imgs.append(resize(edges, (200, 200, 1)))
    labels.append('Alex')
for photo in os.listdir(master_folder + 'Hunter'):
    img = imread(master_folder + '/Hunter/' + photo)
    edges = cv2.Canny(img, 200, 200)
    imgs.append(resize(edges, (200, 200, 1)))
    labels.append('Hunter')

# #### Massage Data

# Take out alpha component of image
# imgs = [img[:,:,[0,1,2]] for img in imgs]

labels = np.array([[1, 0] if label is 'Hunter' else [0, 1]
                   for label in labels])

x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=.5,
                                                    stratify=labels)

x_train = np.stack(x_train, axis=0)
x_test = np.stack(x_test, axis=0)
x_train.shape[-1]

num_classes = 2
# y_train = np_utils.to_categorical(y_train, num_classes)
# y_test = np_utils.to_categorical(y_test, num_classes)

# # Use a generator to feed data because data images are of different dimensions so a numpy array cant
# # be constructed
# def generator(x, y):
#     while True:
#         for i,img in enumerate(x):
#             yield np.expand_dims(img,axis=0), np.expand_dims(y[i],axis=0)


# gen = generator(x_train, y_train)
# val_gen = generator(x_test,y_test)
# num_channels = 3
# num_classes = 2
# data augmentation
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

datagen.fit(x_train)
# gen = generator(x_train, y_train)
# val_gen = generator(x_test,y_test)

batch_size = 4
epochs = 10
num_channels = 1


# ### Model with no transfer learning

model = Sequential()

# Note that we leave the image size as None to allow multiple image sizes
model.add(Conv2D(96, (7, 7), strides=2, padding='same',
                 activation='relu', input_shape=(200, 200, num_channels),
                 name='input_layer'))
model.add(MaxPooling2D((3, 3)))

model.add(Conv2D(48, (5, 5), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D((3, 3)))

model.add(Conv2D(24, (3, 3), strides=1, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(24, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Flatten())
# # Spatial Pooling layer to deal with differing image sizes
# model.add(SpatialPyramidPooling([6, 3, 2,1]))

model.add(Dense(128, activation='relu'))

# Classification layer
model.add(Dense(2, activation='softmax'))

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=15, verbose=1, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0))
# opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


hist = model.fit_generator(datagen.flow(x_train, y_train,
                                        batch_size=batch_size),
                           steps_per_epoch=x_train.shape[0] // batch_size,
                           epochs=epochs,
                           validation_data=(x_test, y_test), verbose=1,
                           callbacks=callbacks)



# ### Utilize transfer learning

filepath = "models/caltech_101_canny_no_pyramid.h5"
trans_model = load_model(filepath)

# Allow no weight adjustments for the pretrained layers
for layer in trans_model.layers:
    layer.trainable = False


# Spatial Pooling layer to deal with differing image sizes
# trans_model.add(SpatialPyramidPooling([1, 2, 4]))

# trans_model.add(Flatten())
# trans_model.add(Conv2D(32,(3,3)))
# trans_model.add(MaxPooling2D((1,1)))
# trans_model.add(Flatten())

trans_model.add(Dense(128, activation='relu'))
# Classification Layer
trans_model.add(Dense(num_classes, activation='softmax'))

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=15, verbose=1, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0))
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
trans_model.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])


epochs = 10
# Fit model
trans_hist = trans_model.fit_generator(datagen.flow(x_train, y_train,
                                                    batch_size=batch_size),
                                       steps_per_epoch=x_train.shape[0] // batch_size,
                                       epochs=epochs,
                                       validation_data=(
                                           x_test, y_test), verbose=1,
                                       callbacks=callbacks)

# Evaluate

trans_pred = []
reg_pred = []
for img in x_test:
    trans_pred.append(trans_model.predict(np.expand_dims(img, axis=0)))
    reg_pred.append(model.predict(np.expand_dims(img, axis=0)))

reg_pred = ["Hunter" if x[0][0] > x[0][1] else "Alex" for x in reg_pred]
trans_pred = ["Hunter" if x[0][0] > x[0][1] else "Alex" for x in trans_pred]

y_test = ["Hunter" if x[0] > x[1] else "Alex" for x in y_test]

print("Model with No transfer learning")
print(classification_report(y_test, reg_pred))
print("Model with transfer learning")
print(classification_report(y_test, trans_pred))
