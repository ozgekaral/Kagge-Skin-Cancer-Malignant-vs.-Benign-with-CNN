# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:11:54 2023

@author: user202
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from  keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import pickle
import cv2
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
from zipfile import ZipFile 

test="test"
train='train'


train_benign = train + '\benign'
train_malignant = train + '\malignant'
test_benign = test + '\benign'
test_malignant = test + '\malignant'

generator = ImageDataGenerator(rescale = 1.0/255.0, zoom_range = 0.2, width_shift_range=0.2,
    height_shift_range=0.2)

training = generator.flow_from_directory(directory =train, target_size = (64, 64), batch_size = 300, class_mode = 'binary')
testing = generator.flow_from_directory(directory = test, 
                                                   target_size = (64, 64),
                                                  batch_size =300,
                                                  class_mode = 'binary')
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training.image_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(filters = 126, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(rate = 0.2))
model.add(Dense(units = len(set(training.classes)), activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


tunning_model = model.fit_generator(training, epochs = 30,
                        validation_data = testing)


def testing_image(image_directory):
    test_image = image.load_img(image_directory, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(x = test_image)
    print(result)
    if result[0][0]  == 1:
        prediction = 'benign'
    else:
        prediction = 'malignant'
    return prediction
print(testing_image(test + '/benign/1.jpg'))