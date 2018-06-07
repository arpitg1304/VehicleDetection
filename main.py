# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:06:16 2017

@author: Arpit

"""
import time
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#There were som =e warnings from keras that needed to be suppressed
import warnings
warnings.filterwarnings("ignore")
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.layers import Flatten

import matplotlib.pyplot as plt

import skimage
from skimage import io


from model import create_model
from test_random_images import test_random
from read_data import read_data
from test_bigger_images import test_image
from find_lanes import laneDetection
from create_video import create_video

print('Showing the results from CNN pipeline -- Deep Learning-- \n')
#Helper function to plot the large size images in a figure
def plot_img(img):
    fig = plt.figure(figsize=(12,20))
    plt.imshow(img)
    plt.show()

#Creating the model using the model architecture defined in the class model


model = create_model(input_shape=(64,64,3))
#model.summary()
model.add(Flatten())

X_train, X_test, Y_train, Y_test = read_data(0.1)

model.compile(loss='mse',optimizer='rmsprop',metrics=['accuracy'])

#Training the model, uncomment the following 2 lines to train the model from scratch

#model.fit(X_train, Y_train, batch_size=256, epochs=1, verbose=2, validation_data=(X_test, Y_test))
#model.save_weights('./dataset/model1.h5')

#Loading the pretrained weights , approx 2 MB of data
model.load_weights('./dataset/model.h5')



#Testing the model on randon validation images that were splitted from the dataset(10%)
#Keep the function call commented if there is no dataset folder

#test_random(X_test, Y_test, model)

#Testing the model for a bigger image(1280x720) from the folder test_images

img = './test_images/test4.jpg'
cars_img, B = test_image(img, False)
#skimage.io.imsave('cars_detected.PNG', cars_img)

#lanes_img = laneDetection(img)

#plot_img(lanes_img)

#Combining the results of both pipelines by passing the image with cars to lanes function
combined_image = laneDetection('cars_detected.PNG')
plot_img(combined_image)

#This call creats the Video by processing the test video frame by frame
#Comment this line for avoiding the video creation
start = time.time()
create_video()
end = time.time()
elapsed = end - start
print(elapsed)

#print('Showing the results from HOG pipeline -- Classical Vision-- \n')

#import VehicleDetectionHog
