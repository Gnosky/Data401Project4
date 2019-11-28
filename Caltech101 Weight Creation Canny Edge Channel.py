#!/usr/bin/env python
# coding: utf-8
# %%
import sys
# sys.path.append('./keras-spp/')
from spp.SpatialPyramidPooling import SpatialPyramidPooling

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
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
import cv2
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(K.tensorflow_backend_get_available_gpus())
# parser = argparse.ArgumentParser()
# parser.add_argument('--epochs',default=10,type=int)
# parser.add_argument('--batch_size',default=32,type=int)


# args = parser.parse_args()
# %% [markdown]
# Load and massage data

# %%
master_caltech_folder = '../GrantData401Project4/101_ObjectCategories/'
object_folders = sorted(os.listdir(master_caltech_folder))

imgs = []
labels = []
count = 0
for i,name in enumerate(object_folders):
    for pic_file in os.listdir(master_caltech_folder+name):
        img = imread(master_caltech_folder+name+"/"+pic_file)
        if len(img.shape) == 3:
            labels.append(i)
            imgs.append(img)
            count += 1
            if count % 25 == 0:
                print('done with', count)

print("num images", len(imgs))

print('perform canny edge detection')

edge_images = []
count = 0
for img in imgs:
    edges = cv2.Canny(img, 300,400)
    edge_images.append(edges)
    count += 1
    if count % 25 == 0:
        print('done with ', count)

# %% [markdown]
# Split into training and testing

# %%
x_train, x_test, y_train, y_test = train_test_split(edge_images, labels, test_size=.2)

num_classes = 102
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# %%
batch_size = 1
num_channels = 1
epochs = 10


model = Sequential()

# Note that we leave the image size as None to allow multiple image sizes
model.add(Conv2D(32, (3, 3), padding='same',
                 activation='relu',input_shape=(None, None,num_channels),
                name='input_layer'))
model.add(Conv2D(32, (3, 3),activation='relu',name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool1'))

model.add(Conv2D(64, (3, 3), padding='same',activation='relu',name='conv3'))
model.add(Conv2D(64, (3, 3),name='conv4'))

# Spatial Pooling layer to deal with differing image sizes
model.add(SpatialPyramidPooling([1, 2, 4]))

# Classification layer
model.add(Dense(num_classes, activation='softmax'))

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=15, verbose=1, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0))
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Use a generator to feed data because data images are of different dimensions so a numpy array cant
# be constructed
def generator(x, y):
    while True:
        for i,img in enumerate(x):
            yield np.expand_dims(img,axis=0), np.expand_dims(y[i],axis=0)

gen = generator(x_train, y_train)
val_gen = generator(x_test,y_test)

# Fit model
hist = model.fit_generator(generator=gen,
                steps_per_epoch = 1,
                epochs=1,validation_data=val_gen,
                validation_steps=len(x_test))

# %%
print("summary before removing layer")
print(model.summary())

model.pop() # remove dense layer
model.pop() # Remove Spatial Layer
print("summary after removing layer")

print(model.summary())


# %%
"""
Save model for usage in AlexHunter
"""
model.build(None)
model.save("caltech_101_canny_edge_channel.h5")
