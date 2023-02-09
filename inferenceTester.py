from simpleNetwork import CustomBiggerNet
import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import cv2
from glob import glob
from os import path
import matplotlib.pyplot as plt 
import torch.optim as optim
import torch
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
# from networkTrainer import *
from HarrysLibs import *


#~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
pygame.init()
pygame.display.set_mode((300,300)) #size of pop-up window
pygame.key.set_repeat(100) #holding a key sends continuous KEYDOWN events. Input argument is milli-seconds delay between events and controls the sensitivity
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# stop the robot 
ppi.set_velocity(0,0)
print("initialise camera")
camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')
INNER_WHEEL = 25
OUTER_WHEEL = 35



if __name__=="__main__":
    #~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
    print("setting up")
    pygame.init()
    pygame.display.set_mode((300,300)) #size of pop-up window
    pygame.key.set_repeat(100) #holding a key sends continuous KEYDOWN events. Input argument is milli-seconds delay between events and controls the sensitivity
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # stop the robot 
    ppi.set_velocity(0,0)

    # init
    print("initialise camera")
    camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')
    time.sleep(2)
    print(camera)
    INNER_WHEEL = 25
    OUTER_WHEEL = 35

    #Load the model from file
    print("using preloaded model")
    #Load the model from file
    net = CustomBiggerNet()
    net.load_state_dict(torch.load('model'))
    net.eval()

    #Consts for getting outputs OPPOSITE from training
    LEFT = 2
    STRAIGHT = 1
    RIGHT = 0

    #Speed consts
    scale = 1
    FANGIN = 20*scale
    INNER_TURN = 0 *scale
    OUTER_TURN = 20 *scale
    INNER_ADJ = 10 *scale
    OUTER_ADJ = 15 *scale
    STRAIGHT_ADJ = 15 *scale

    steerConf = 0
    steerTracker = 4
    steerThresh = 3
    trackConf = 0
    trackTracker = 4

    try:
        # MAIN LOOP
        while True:
            #get image
            image = camera.frame
            #set controls
            steer, track = modelPredictSteerClass(image, net)
            
            #Check confidence
            if trackTracker == track:
                trackConf += 1
            else:
                trackConf = 0
                trackTracker = track
            if steerTracker == steer:
                steerConf += 1
            else:
                steerConf = 0
                steerTracker = steer
            
            # steerAdd = trackConf 
            
            #If steer conf is high, change the steer
            if steerConf > steerThresh:
                print("CHANGE STEER")
                #Cornering
                if steer == LEFT and track == LEFT:
                    ppi.set_velocity(INNER_TURN, OUTER_TURN+steerConf) 
                    print("turning left on left")
                elif steer == RIGHT and track == RIGHT:
                    ppi.set_velocity(OUTER_TURN, INNER_TURN+steerConf) 
                    print("turning right on right")

                #Fanging
                elif steer == STRAIGHT and track == STRAIGHT:
                    ppi.set_velocity(FANGIN+steerConf, FANGIN+steerConf) 
                    print("FANGING")

                #Adjusting
                elif steer == LEFT and track != LEFT:
                    ppi.set_velocity(INNER_ADJ, OUTER_ADJ+steerConf) 
                    print("Adjusting left")
                elif steer == RIGHT and track != RIGHT:
                    ppi.set_velocity(OUTER_ADJ+steerConf, INNER_ADJ) 
                    print("Adjusting right")
                elif steer == STRAIGHT and track != STRAIGHT:
                    ppi.set_velocity(STRAIGHT_ADJ+steerConf, STRAIGHT_ADJ+steerConf) 
                    print("adjusting straight")
                else:
                    raise Exception ("Wrong combo of steer and track")


            # SPACE for shutdown 
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print("stop")                    
                        ppi.set_velocity(0,0)
                        raise KeyboardInterrupt
    #stops motors on shutdown
    except KeyboardInterrupt:
        ppi.set_velocity(0,0)