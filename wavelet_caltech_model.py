#!/usr/bin/env python
# coding: utf-8
# %%
import sys
# sys.path.append('./keras-spp/')
# from spp.SpatialPyramidPooling import SpatialPyramidPooling

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D, GaussianNoise
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD
import pywt
import numpy as np
import glob
import os
from imageio import imread
from skimage.transform import resize
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2gray
from keras import backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split


import argparse
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(K.tensorflow_backend_get_available_gpus())
# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs',default=10,type=int)
# parser.add_argument('--batch_size',default=32,type=int)


# args = parser.parse_args()
# %% [markdown]
# Load and massage data

# %%
master_caltech_folder = '../../Desktop/101_ObjectCategories/'
object_folders = sorted(os.listdir(master_caltech_folder))

imgs = []
labels = []
for i,name in enumerate(object_folders):
    for pic_file in os.listdir(master_caltech_folder+name):
        img = imread(master_caltech_folder+name+"/"+pic_file)
        if len(img.shape) == 3:
            labels.append(i)
            imgs.append(resize(img,(200,200,3)))

print("num images", len(imgs))

# %% [markdown]
# Split into training and testing

# %%

wavelet_imgs = []
for i, img in enumerate(imgs):
    if i % 50 == 0:
        print(i)
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    wavelet_imgs.append(coeffs2[1][:2])

for i,stack in enumerate(wavelet_imgs):
    stack = np.dstack(stack)
    wavelet_imgs[i] = stack


x_train, x_test, y_train, y_test = train_test_split(wavelet_imgs, labels, test_size=.1)
x_train = np.stack(x_train,axis=0)
x_test = np.stack(x_test,axis=0)

num_classes = 102
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


print(x_train.shape, x_test.shape)
# %%

num_channels = 8

# # Note that we leave the image size as None to allow multiple image sizes
# model.add(Conv2D(32, (3, 3), padding='same',
#                  activation='relu',input_shape=(None, None,num_channels),
#                 name='input_layer'))
# model.add(Conv2D(32, (3, 3),activation='relu',name='conv2'))
# model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1'))
          
# model.add(Conv2D(64, (3, 3), padding='same',activation='relu',name='conv3'))
# model.add(Conv2D(64, (3, 3),name='conv4'))

# # Spatial Pooling layer to deal with differing image sizes
# model.add(SpatialPyramidPooling([1, 2, 4]))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                 input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                 input_shape=(x_train.shape[1],x_train.shape[2],1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                 input_shape=(x_train.shape[1],x_train.shape[2],1)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
#                  input_shape=(x_train.shape[1],x_train.shape[2],1)))
# model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
# model.add(MaxPooling2D((2, 2)))
# model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))

opt = SGD(lr=.0001, momentum=.9)

model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])

# %%
# Use a generator to feed data because data images are of different dimensions so a numpy array cant
# be constructed
# def generator(x, y):
#     while True:
#         for i,img in enumerate(x):
#             yield np.expand_dims(img,axis=0), np.expand_dims(y[i],axis=0)
#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=True,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=True,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)
# gen = generator(x_train, y_train)
# val_gen = generator(x_test,y_test)

EPOCHS = 10
BATCH = 8

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                   patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))

hist = model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=BATCH),
                    steps_per_epoch = x_train.shape[0] // BATCH,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),verbose=1,
                    callbacks = callbacks)
# %%
print("summary before removing layer")
print(model.summary())


# %%
model.pop() # remove dense layer
model.pop() # Remove dense Layer
model.pop() # Remove flatten layer
model.pop()
print("summary after removing layer")

print(model.summary())


# %%
"""
Save model for usage in AlexHunter
"""
model.build(None)
model.save("wavelet_model.h5")

