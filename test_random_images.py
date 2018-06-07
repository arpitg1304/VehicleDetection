# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 20:48:09 2017

@author: Arpit
"""
import numpy as np
import matplotlib.pyplot as plt

def test_random(X_test, Y_test, model):
    i = np.random.randint(len(X_test))
    plt.imshow(X_test[i])
    reshaped = np.reshape(X_test[i], (1,64,64,3))
    prediction = model.predict(reshaped, batch_size = 64, verbose = 2)
    
    if prediction[0][0] >= 0.5:
        print("There is a car in the Image with probability: " + str(prediction))
    else:
        print("There is no car in the Image with probability: " + str(prediction))
    
    print('\n')
    grnd_truth = Y_test[i]
    if grnd_truth == 1:
        print('There is actually a car in the image\n')
    else:
        print('There is no car in the image')
    
    
    
    