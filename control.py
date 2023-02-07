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

#~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
pygame.init()
pygame.display.set_mode((300,300)) #size of pop-up window
pygame.key.set_repeat(100) #holding a key sends continuous KEYDOWN events. Input argument is milli-seconds delay between events and controls the sensitivity
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# stop the robot 
ppi.set_velocity(0,0)
print("initialise camera")
camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')


try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    print("right")
                    ppi.set_velocity(20,20)
                if event.key == pygame.K_RIGHT:
                    print("right")
                    ppi.set_velocity(10,0)
                if event.key == pygame.K_LEFT:
                    print("left")
                    ppi.set_velocity(0,10)
                if event.key == pygame.K_SPACE:
                    print("stop")                    
                    ppi.set_velocity(0,0)
                    raise KeyboardInterrupt
            if event.type == pygame.KEYUP:
                ppi.set_velocity(0,0)

        image = camera.frame

        
        

        
except KeyboardInterrupt:    
    ppi.set_velocity(0,0)