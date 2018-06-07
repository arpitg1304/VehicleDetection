# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:16:14 2017

@author: Arpit
"""
from sklearn.model_selection import train_test_split
import glob
import cv2
import numpy as np

def read_data(test_size):
    cars = glob.glob("./dataset/vehicles/*/*.png")
    non_cars = glob.glob("./dataset/non-vehicles/*/*.png")
    
    data = []
    for file in cars:    
        data.append(cv2.imread(file))
        
    for file in non_cars:    
        data.append(cv2.imread(file))
    data = np.array(data)
    
    # Generate Y Vector
    Y_cars = np.ones(len(cars))
    Y_non_cars = np.zeros(len(non_cars))
    Y = np.concatenate([Y_cars, Y_non_cars])
    
    # Split train and validation dataset using sklearn's train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size=test_size, random_state=63)
    return X_train, X_test, Y_train, Y_test

