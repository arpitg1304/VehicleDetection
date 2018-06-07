# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 22:27:39 2017

@author: Arpit
"""
import skimage
from skimage import io
import matplotlib.pyplot as plt
from car_scanner import scan_car


def test_image(img, isShow):
    test_img = skimage.io.imread(img)
    if isShow:
        fig = plt.figure(figsize=(12,20))
        plt.imshow(test_img)
        plt.show()
    cars_img, B = scan_car(img, isShow)
    return cars_img, B
