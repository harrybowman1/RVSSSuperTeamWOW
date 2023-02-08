#!/usr/bin/env python3
import time
import click
import math
import sys
sys.path.append("..")
import cv2
import numpy as np
import penguinPi as ppi
import pygame

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
    #~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
    pygame.init()
    pygame.display.set_mode((300,300)) #size of pop-up window
    pygame.key.set_repeat(100) #holding a key sends continuous KEYDOWN events. Input argument is milli-seconds delay between events and controls the sensitivity
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # stop the robot 
    ppi.set_velocity(0,0)
    print("initialise camera")
    camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')
    time.sleep(2)
    print(camera)
    INNER_WHEEL = 25
    OUTER_WHEEL = 35

    try:
        while True:
            image = camera.frame
            command = steer_away_from_green(image)
            if command == "LEFT":
                ppi.set_velocity(INNER_WHEEL, OUTER_WHEEL) 
            elif command == "RIGHT":
                ppi.set_velocity(OUTER_WHEEL, INNER_WHEEL) 
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print("stop")                    
                        ppi.set_velocity(0,0)
                        raise KeyboardInterrupt

    except KeyboardInterrupt:
        ppi.set_velocity(0,0)


        


        
