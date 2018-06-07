# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 03:08:02 2017

@author: Arpit
"""
from moviepy.editor import VideoFileClip
from find_lanes import laneDetection
from test_bigger_images import test_image
import numpy as np
from collections import deque
import cv2
import skimage
from skimage import io

# video processing pipeline
def process_video(img):

    history = deque(maxlen=30)

    skimage.io.imsave('c2.PNG', img)
    im2, boxes = test_image('c2.PNG', False)

    skimage.io.imsave('c1.PNG', im2)

    img_lanes = laneDetection('c1.PNG')



    # Iterate through all detected cars

    for b in range(1, boxes[1]+1):
        # Find pixels with each car_number label value
        nonzero = (boxes[0] == b).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Append current boxe to history

        history.append([np.min(nonzerox),np.min(nonzeroy),np.max(nonzerox),np.max(nonzeroy)])

    # Get recent boxes for the last 30 fps
    r_boxes = np.array(history).tolist()

    # Groups the object candidate rectangles with difference of 10%
    boxes = cv2.groupRectangles(r_boxes, 10, .1)

    # Draw rectangles if found
    if len(boxes[0]) != 0:
        for box in boxes[0]:
            cv2.rectangle(img_lanes, (box[0], box[1]), (box[2],box[3]), (0,0,255), 6)

    # Return image with found cars and lanes
    return img_lanes
def create_video():
    clip_output = 'output_videos/test_video.mp4'
    clip = VideoFileClip("test_videos/test_video.mp4")
    clip_process = clip.fl_image(process_video)
    clip_process.write_videofile(clip_output, audio=False)
