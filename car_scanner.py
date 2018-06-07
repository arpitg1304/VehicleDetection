# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:57:10 2017

@author: Arpit
"""
from model import create_model
import matplotlib.pyplot as plt
from skimage import io
import skimage
import numpy as np
import cv2

from thresholding import heat_n_thresholding

def scan_car(img, isShow):
    color=(0, 0, 255)
    thickness=6
    thres = 3
    #Using Transfer Learning to use the pretrained model on diferent images
    model = create_model(input_shape=(260,1280,3))
    model.load_weights('./dataset/model.h5')
    img = skimage.io.imread(img)
    crop = img[400:660, 0:1280]
    crop_reshaped = crop.reshape(1,crop.shape[0],crop.shape[1],crop.shape[2])
    H = model.predict(crop_reshaped)
    xx, yy = np.meshgrid(np.arange(H.shape[2]),np.arange(H.shape[1]))
    x = (xx[H[0,:,:,0]>0.99999])
    y = (yy[H[0,:,:,0]>0.99999])
    hot_sectors = []
    # We save those rects in a list
    for i,j in zip(x,y):
        hot_sectors.append(((i*8,400 + j*8), (i*8+64,400 +j*8+64)))

    copy = np.copy(img)

    for hots in hot_sectors:
        cv2.rectangle(copy, hots[0], hots[1], color, thickness)

    if isShow:
        fig = plt.figure(figsize=(12,20))
        plt.imshow(copy)
        plt.show()

    cars_img, B = heat_n_thresholding(img, hot_sectors, thres, isShow)
    return cars_img, B
