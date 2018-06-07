# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 01:41:33 2017

@author: Arpit
"""
import numpy as np
from scipy.ndimage.measurements import label
import cv2
import matplotlib.pyplot as plt

def heat_n_thresholding(img, hot_sectors, thres, isShow):
    heat =  np.zeros_like(img[:,:,0]).astype(np.float)
    
    for box in hot_sectors:
        heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    heat[heat <= thres] = 0
    
    heat = np.clip(heat, 0, 255)
    
    B = label(heat)
    
    copy1 = np.copy(img)
    
    for n in range(1, B[1]+1):
        # Find pixels with each car_number label value
        nonzero = (B[0] == n).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(copy1, bbox[0], bbox[1], (0,0,255), 4)
    
    if isShow:
        fig = plt.figure(figsize=(12,20))
        plt.imshow(copy1)
        plt.show()
        print(str(B[1]) + ' Cars have been found in the image')
    
    return copy1, B
    
    
    
    
    
    