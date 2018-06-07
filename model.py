# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:57:43 2017

@author: Arpit
"""

from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Lambda


def create_model(input_shape=(64,64,3)):
    model = Sequential()
    #normalizing the data
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    # 1st conv layer with 64 filter, 3x3 each, 60% dropout
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))
    model.add(Dropout(0.6))
    # 2nd conv layer with 64 filter, 3x3 each, 60% dropout
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv2',border_mode="same"))
    model.add(Dropout(0.6))
    # 3rd conv layer with 64 filter, 3x3 each, 60% dropout
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv3',border_mode="same"))
    model.add(Dropout(0.6))
    # 4th conv layer with 64 filter, 3x3 each, 8x8 pooling and dropout
    model.add(Conv2D(64, 3, 3, activation='relu', name='conv4',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.6))
    # Fully connected layer
    model.add(Conv2D(64,8,8,activation="relu",name="dense1"))
    model.add(Dropout(0.6))
    # Output layer with sigmoid activation
    model.add(Conv2D(1,1,1,name="dense2", activation="sigmoid"))
    # model.summary()
    return model
