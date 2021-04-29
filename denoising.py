# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:28:00 2020

@author: rejid4996
"""

# loading packages
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
import cv2
import keras
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf

from sklearn.model_selection import train_test_split

#!nvidia-smi

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from IPython.display import SVG
%matplotlib inline

path = r'C:\Users\rejid4996\OneDrive - ARCADIS\Arcadis Files\Project Files\Personal Projects\Autoencoders'

# store image names in list for later use
train_img = sorted(os.listdir(path + '/train/train'))
train_cleaned_img = sorted(os.listdir(path + '/train_cleaned/train_cleaned'))
test_img = sorted(os.listdir(path + '/test/test'))




## data preparation
#os.chdir(r'C:\Users\rejid4996\OneDrive - ARCADIS\Arcadis Files\Project Files\Autoencoders\train\train')
#test_path = r'C:\Users\rejid4996\OneDrive - ARCADIS\Arcadis Files\Project Files\Autoencoders\testing'

def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (540, 420))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (420, 540, 1))
    
    return img


# preprocess images
train = []
train_cleaned = []
test = []

for f in sorted(os.listdir(path + '\\train\\train\\')):
    train.append(process_image(path + '\\train\\train\\' + f))

for f in sorted(os.listdir(path + '\\train_cleaned\\train_cleaned\\')):
    train_cleaned.append(process_image(path + '\\train_cleaned\\train_cleaned\\' + f))
   
for f in sorted(os.listdir(path + '\\test\\test\\')):
    test.append(process_image(path + '\\test\\test\\' + f))

##
# convert list to numpy array
X_train = np.asarray(train)
Y_train = np.asarray(train_cleaned)
X_test = np.asarray(test)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)

def model():
    input_layer = Input(shape=(420, 540, 1))  # we might define (None,None,1) here, but in model summary dims would not be visible
    
    # encoding
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Dropout(0.5)(x)

    # decoding
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)

    output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])

    return model

model = model()
model.summary()

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model,show_shapes=True, dpi=70).create(prog='dot', format='svg'))

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#callback = EarlyStopping(monitor='loss', patience=30)
history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=600, batch_size=24, verbose=1)

#%%
import keras.backend as K


with tf.Session(config=config) as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    callback = EarlyStopping(monitor='loss', patience=30)
    history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=600, batch_size=24, verbose=1, callbacks=[callback])
