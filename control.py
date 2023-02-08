#!/usr/bin/env python3
import time
import click
import math
import sys
sys.path.append("..")
import cv2
import numpy as np
import penguinPi as ppi

def steer_away_from_green(img: np.ndarray) -> str:
    """returns a driving command to drive away from the centroid of the green channel of img
    
    Args:
        img: imput image
        
    Returns:
        either 'LEFT' or 'RIGHT'
    """

    assert img is not None, "no image file found"

    moments = cv2.moments(img[:, :, 1])
    x_centroid = moments['m10'] / moments['m00']
    y_centroid = moments['m01'] / moments['m00']

    (height, width, channels) = img.shape
    assert y_centroid <= height
    assert x_centroid <= width
    assert y_centroid >= 0
    assert y_centroid >= 0
    
    if x_centroid < width/2:
        return "RIGHT"

    return "LEFT"

if __name__=="__main__":
    # stop the robot 
    ppi.set_velocity(0,0)
    print("initialise camera")
    camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')
    print(camera)
    INNER_WHEEL = 25
    OUTER_WHEEL = 35
    while True:
        image = camera.frame
        command = steer_away_from_green(image)
        if command == "LEFT":
            ppi.set_velocity(INNER_WHEEL, OUTER_WHEEL) 
        elif command == "RIGHT":
            ppi.set_velocity(OUTER_WHEEL, INNER_WHEEL) 

        


        
