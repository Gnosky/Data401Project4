#!/usr/bin/env python
# coding: utf-8
# %%

# TODO:
#  - find ways to speed up training process (perhaps use keras flow_from_directory)
#  - figure out what to do with different size images
#  - Experiment with different architectures

# %%


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


import argparse
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#print(K.tensorflow_backend_get_available_gpus())
parser = argparse.ArgumentParser()
parser.add_argument('--epochs',default=10,type=int)
parser.add_argument('--batch_size',default=32,type=int)


args = parser.parse_args()
# %%


master_caltech_folder = '101_ObjectCategories/'
object_folders = sorted(os.listdir(master_caltech_folder))


# #### Load images

# %%


imgs = []
labels = []
for i,name in enumerate(object_folders):
    for pic_file in os.listdir(master_caltech_folder+name):
        img = imread(master_caltech_folder+name + "/"+pic_file)
        if len(img.shape) == 3:
            labels.append(i)
            imgs.append(resize(rgb2gray(img),(300,400)).astype(np.float))

print("Data loaded")
print("num images", len(imgs))

# #### Split into training and testing

# %%
# Apply a 2d wavelet transformation to each image

print("applying wavelet transform")
wavelet_imgs = []
for i, img in enumerate(imgs):
	if i % 1000 == 0:
		print("On image",i)
	coeffs2 = pywt.dwt2(img, 'bior1.3')
	horiz_vert = coeffs2[1][:2]
	horiz_vert = np.dstack(horiz_vert)
	# Store horizontal detail
	wavelet_imgs.append(horiz_vert)


seed = 7
np.random.seed(seed)

#x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size = 0.25)
x_train, x_test, y_train, y_test = train_test_split(wavelet_imgs, labels, test_size = 0.25)

x_train = np.stack(x_train, axis=0)
y_train = np.stack(y_train, axis=0)
x_test = np.stack(x_test, axis=0)
y_test = np.stack(y_test, axis=0)

num_classes = 102

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

print("x_train.shape",x_train.shape)
print("x_test.shape",x_test.shape)
print("y_train.shape",y_train.shape)
print("y_test.shape",y_test.shape)
"""
np.save('x_train',x_train)
np.save('x_test',x_test)
np.save('y_train',y_train)
np.save('y_test',y_test)

print("Data saved")
"""
# #### Create data generator for input to network

# %%


#x_train = np.expand_dims(x_train, axis=3)
#x_test = np.expand_dims(x_test, axis=3)


#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True)  # randomly flip images

datagen.fit(x_train)


# #### Build and fit network

# %%
x_shape = (wavelet_imgs[0].shape[0],wavelet_imgs[1].shape[1],wavelet_imgs[2].shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=x_shape))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())



model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(num_classes, activation='softmax'))

opt = SGD(lr=0.0001, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['accuracy'])


# %%
EPOCHS = args.epochs
BATCH = args.batch_size

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=15, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0))
callbacks.append(ModelCheckpoint(filepath='full_caltech_101.h5',monitor='val_loss', save_weights_only=True,save_best_only=True))

hist = model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=BATCH),
                    steps_per_epoch=x_train.shape[0] // BATCH,
                    epochs=EPOCHS,
                    validation_data=(x_test, y_test),verbose=1,
                    callbacks=callbacks)


# #### Remove the last dense layer for transfer learning's sake

# %%
model.summary()

# %%
model.pop() # remove dense layer
model.pop() # Remove dense layer
model.pop() # Remove flatten layer
print(model.summary())


# #### Save weights for use in AlexHunterNet

# %%
model.build(None)
model.save("caltech_101.h5")

