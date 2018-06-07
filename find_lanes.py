# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 03:17:44 2017

@author: Arpit
"""

# Import libraries
import numpy as np
import cv2
import skimage
from   skimage import io

# Function for lane detection
def laneDetection(image):
    # Read image and crop it
    image     = skimage.io.imread(image)
    cropdImg  = np.copy(image)
    cropdImg  = regionOfInterest(cropdImg, [srcPoints.astype(np.int32)])
    # Image transformation
    warpedImg = imageTransformation(cropdImg)
    # Sobel and color mask on the image
    sobelImg  = imageSobelMask(warpedImg)
    colorImg  = imageColorMask(warpedImg)
    # Combine Color and sobel mask 
    maskImg   = combColorSobelMasks(sobelImg, colorImg)
    # Find the lines from polyfit
    leftFit, rightFit, _ = slidingWindow(maskImg)
    # Create the lane mask and apply backtransformation
    laneMask  = applyBackTrans(image, leftFit, rightFit)
    # Combine the sample image with the lane layer
    imgResult = cv2.addWeighted(image, 1, laneMask, 1, 0)
    return imgResult 

# Define the source points
srcPoints = np.float32([[0 , 720],
                         [1280 , 720],
                         [750 , 470],
                         [530 , 470]])

# Define the destination points
dstPoints = np.float32([[320 , 720],
                         [960 , 720],
                         [960 , 0],
                         [320 , 0]])
# Storing averages
prevFrames = []

# Calculate Region Of Interest
def regionOfInterest(img,vertices):
    # Defining a blank mask 
    mask = np.zeros_like(img)   

    # Defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channelCount = img.shape[2]  
        ignoreMaskColor = (255,) * channelCount
    else:
        ignoreMaskColor = 255
        
    # Filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignoreMaskColor) 
    # Returning the image only where mask pixels are nonzero
    maskedImage = cv2.bitwise_and(img, mask)
    return maskedImage

def imageTransformation(img):
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
    # Warp the image using OpenCV warpPerspective()
    transformed = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # Return transformed image
    return transformed

def absSobelThresh(img, orient='x', threshMin=0, threshMax=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        absSobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    if orient == 'y':
        absSobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))
    # Rescale back to 8 bit integer
    scaledSobel = np.uint8(255*absSobel/np.max(absSobel))
    # Create a copy and apply the threshold
    binaryOutput = np.zeros_like(scaledSobel)
    # Using Inclusive (>=, <=) thresholds
    binaryOutput[(scaledSobel >= threshMin) & (scaledSobel <= threshMax)] = 1
    return binaryOutput

def magThresh(img, threshMin=0, threshMax=255):
    # Take both Sobel x and y gradients
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=9)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=9)
    # Calculate the gradient magnitude
    gradMag = np.sqrt(sobelX**2 + sobelY**2)
    # Rescale to 8 bit
    scaleFactor = np.max(gradMag)/255 
    gradMag = (gradMag/scaleFactor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binaryOutput = np.zeros_like(gradMag)
    binaryOutput[(gradMag >= threshMin) & (gradMag <= threshMax)] = 1
    # Return the binary image
    return binaryOutput

def imageSobelMask(img):
    # Convert to HLS and extract L and S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    lChannel = hls[:,:,1]
    sChannel = hls[:,:,2]
    # Apply sobel in x direction on L and S channel
    lChannelSobelX = absSobelThresh(lChannel,'x', 20, 200)
    sChannelSobelX = absSobelThresh(sChannel,'x', 60, 200)
    sobelCombinedX = cv2.bitwise_or(sChannelSobelX, lChannelSobelX)    
    # Apply magnitude sobel
    lChannelMag = magThresh(lChannel, 80, 200)
    sChannelMag = magThresh(sChannel, 80, 200)
    magCombined = cv2.bitwise_or(lChannelMag, sChannelMag)   
    # Combine all the sobel filters
    maskCombined = cv2.bitwise_or(magCombined, sobelCombinedX)    
    # Mask out the desired image and filter image again
    maskCombined = regionOfInterest(maskCombined, np.array([[(330, 0),(950, 0), (950, 680), (330, 680)]]))   
    # Return the sobel mask
    return maskCombined

def imageColorMask(img):
    # Convert to HLS and extract S and V channel
    imgHsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Define color thresholds in HSV
    whiteLow = np.array([[[0, 0, 210]]])
    whiteHigh = np.array([[[255, 30, 255]]])
    yellowLow = np.array([[[18, 80, 80]]])
    yellowHigh = np.array([[[30, 255, 255]]])
    # Apply the thresholds to get only white and yellow
    whiteMask = cv2.inRange(imgHsv, whiteLow, whiteHigh)
    yellowMask = cv2.inRange(imgHsv, yellowLow, yellowHigh)
    # Bitwise or the yellow and white mask
    colorMask = cv2.bitwise_or(yellowMask, whiteMask)
    return colorMask

# Combine Sobel and Color Masks
def combColorSobelMasks(sobelMask, colorMask):
    maskCombined = np.zeros_like(sobelMask)
    maskCombined[(colorMask>=.5)|(sobelMask>=.5)] = 1
    return maskCombined

# Function to calculate Window Mask
def windowMask(width, height, imgRef, center,level):
    output = np.zeros_like(imgRef)
    output[int(imgRef.shape[0]-(level+1)*height):int(imgRef.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),imgRef.shape[1])] = 1
    return output

# Function to calcualte Sliding Window
def slidingWindow(img):
    # Window settings
    windowWidth  = 50
    windowHeight = 100
    # How much to slide left and right for searching
    margin = 30    
    # Store the (left,right) window centroid positions per level
    windowCentroids = [] 
    # Create the window template that is used for convolutions
    window = np.ones(windowWidth)     
    # Find the starting point for the lines
    lSum    = np.sum(img[int(3*img.shape[0]/5):,:int(img.shape[1]/2)], axis=0)
    lCenter = np.argmax(np.convolve(window,lSum))-windowWidth/2
    rSum    = np.sum(img[int(3*img.shape[0]/5):,int(img.shape[1]/2):], axis=0)
    rCenter = np.argmax(np.convolve(window,rSum))-windowWidth/2+int(img.shape[1]/2)    
    windowCentroids.append((lCenter,rCenter))
    
    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(img.shape[0] / windowHeight)):
        # convolve the window into the vertical slice of the image
        imageLayer = np.sum(img[int(img.shape[0]-(level+1)*windowHeight):int(img.shape[0]-level*windowHeight),:], axis=0)
        convSignal = np.convolve(window, imageLayer)
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset     = windowWidth / 2
        # Find the best left centroid by using past left center as a reference
        lMinIndex  = int(max(lCenter+offset-margin,0))
        lMaxIndex  = int(min(lCenter+offset+margin,img.shape[1]))
        lCenter    = np.argmax(convSignal[lMinIndex:lMaxIndex])+lMinIndex-offset
        # Find the best right centroid by using past right center as a reference
        rMinIndex  = int(max(rCenter+offset-margin,0))
        rMaxIndex  = int(min(rCenter+offset+margin,img.shape[1]))
        rCenter    = np.argmax(convSignal[rMinIndex:rMaxIndex])+rMinIndex-offset
        windowCentroids.append((lCenter,rCenter))
    
    # If found any window centers, print error and return
    if len(windowCentroids) == 0:
        print("No windows found in this frame!")
        return
    
    # Points used to draw all the left and right windows
    lPoints = np.zeros_like(img)
    rPoints = np.zeros_like(img)

    # Go through each level and draw the windows
    for level in range(0,len(windowCentroids)):
        # Window_mask is a function to draw window areas
        lMask = windowMask(windowWidth,windowHeight,img,windowCentroids[level][0],level)
        rMask = windowMask(windowWidth,windowHeight,img,windowCentroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        lPoints[(lPoints == 255) | ((lMask == 1) ) ] = 255
        rPoints[(rPoints == 255) | ((rMask == 1) ) ] = 255

    # Draw the results
    template = np.array(rPoints+lPoints,np.uint8) # add both left and right window pixels together
    zeroChannel = np.zeros_like(template) # create a zero color channle 
    template = np.array(cv2.merge((template, template, template)),np.uint8) # make window pixels green
    warpage = np.array(cv2.merge((img, img, img)),np.uint8) # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    
    # Extract left and right line pixel positions
    leftx = np.nonzero(lPoints)[1]
    lefty = np.nonzero(lPoints)[0]
    rightx = np.nonzero(rPoints)[1]
    righty = np.nonzero(rPoints)[0]
            
    # Fit a second order polynomial to each
    leftFit = np.polyfit(lefty, leftx, 2)
    rightFit = np.polyfit(righty, rightx, 2)  
    # Return left and right lines as well as the image
    return leftFit, rightFit, output

def applyBackTrans(img, leftFit, rightFit):
    plotY = np.linspace(0, 719, num=720)
    # Calculate left and right x positions
    leftFitX  = leftFit[0]*plotY**2 + leftFit[1]*plotY + leftFit[2]
    rightFitX = rightFit[0]*plotY**2 + rightFit[1]*plotY + rightFit[2]   
    # Defining a blank mask to start with
    polygon = np.zeros_like(img) 

    # Create an array of points for the polygon
    plotY    = np.linspace(0, img.shape[0]-1, img.shape[0])
    ptsLeft  = np.array([np.transpose(np.vstack([leftFitX, plotY]))])
    ptsRight = np.array([np.flipud(np.transpose(np.vstack([rightFitX, plotY])))])
    pts      = np.hstack((ptsLeft, ptsRight))

    # Draw the polygon in blue
    cv2.fillPoly(polygon, np.int_([pts]), (0, 0, 255))
    # Calculate top and bottom distance between the lanes
    topDist    = rightFitX[0] - leftFitX[0]
    bottomDist = rightFitX[-1] - leftFitX[-1]
    
    # Add the polygon to the list of last frames if it makes sense
    if len(prevFrames) > 0: 
        if topDist < 300 or bottomDist < 300 or topDist > 500 or bottomDist > 500:
            polygon = prevFrames[-1]
        else:
            prevFrames.append(polygon)
    else:
        prevFrames.append(polygon)
        
    # Check that the new detected lane is similar to the one detected in the previous frame
    polygonGray = cv2.cvtColor(polygon, cv2.COLOR_RGB2GRAY) 
    prevGray = cv2.cvtColor(prevFrames[-1], cv2.COLOR_RGB2GRAY)  
    nonSimilarity = cv2.matchShapes(polygonGray,prevGray, 1, 0.0)
    if nonSimilarity > 0.002: 
        polygon = prevFrames[-1]

    # Calculate the inverse transformation matrix
    mInv = cv2.getPerspectiveTransform(dstPoints, srcPoints)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    imageBacktrans = cv2.warpPerspective(polygon, mInv, (img.shape[1], img.shape[0]))     
    # Return the 8-bit mask
    return np.uint8(imageBacktrans)
