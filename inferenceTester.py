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


def smooth_command(history):
    """returns most likely command along with the number of occurences"""
    histogram = [0, 0, 0]
    for elem in history:
        histogram[elem] = histogram[elem] + 1
    confidence = max(histogram)
    best_command = max(zip(histogram, range(len(histogram))))[1]
    return best_command, confidence

def ramp(value, value_prev, slope):
    if value > value_prev:
        return min(value, value_prev+slope)
    elif value < value_prev:
        return max(value, value_prev-slope)
    return value
        

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
    scale = 0.2
    FANGIN = int(20*scale)
    INNER_TURN = int(0 *scale)
    OUTER_TURN = int(20 *scale)
    INNER_ADJ = int(10 *scale)
    OUTER_ADJ = int(15 *scale)
    STRAIGHT_ADJ = int(15 *scale)
    ACCELERATION = 5

    steerConf = 0
    steerTracker = 4
    steerThresh = 3
    trackConf = 0
    trackTracker = 4

    state = 4
    n_history_keep = 5
    steer_history = []
    track_history = []

    left_perv, right_prev = (0, 0)

    try:
        # MAIN LOOP
        steerVals = []
        trackVals = []
        while True:
            #get image
            image = camera.frame
            #set controls
            steer, track = modelPredictSteerClass(image, net)
            track_history.append(track)
            steer_history.append(steer)
            if len(steer_history) > n_history_keep:
                del track_history[0]
                del steer_history[0]

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
            
            smoothed_steer, smoothed_steer_conf = smooth_command(steer_history)
            smoothed_track, _ = smooth_command(track_history)
            
            #If steer conf is high, change the steer
            if smoothed_steer_conf > steerThresh:
                # steerVals.append(steer)
                # trackVals.append(track)
                #Cornering
                if smoothed_steer == LEFT and smoothed_track == LEFT:
                    left, right = INNER_TURN, OUTER_TURN
                    # print("turning left on left")
                elif steer == RIGHT and smoothed_track == RIGHT:
                    left,right = OUTER_TURN, INNER_TURN
                    # print("turning right on right")

                #Fanging
                elif smoothed_steer == STRAIGHT and smoothed_track == STRAIGHT:
                    left,right = FANGIN, FANGIN 
                    # print("FANGING")

                #Adjusting
                elif smoothed_steer == LEFT and smoothed_track != LEFT:
                    left, right = INNER_ADJ, OUTER_ADJ 
                    # print("Adjusting left")
                elif smoothed_steer == RIGHT and smoothed_track != RIGHT:
                    left, right = OUTER_ADJ, INNER_ADJ 
                    # print("Adjusting right")
                elif smoothed_steer == STRAIGHT and smoothed_track != STRAIGHT:
                    left, right = STRAIGHT_ADJ, STRAIGHT_ADJ 
                    # print("adjusting straight")
                else:
                    raise Exception ("Wrong combo of steer and track")

                if smoothed_steer == LEFT:
                    steerWord = "LEFT"
                elif smoothed_steer == RIGHT:
                    steerWord = "RIGHT"
                elif smoothed_steer == STRAIGHT:
                    steerWord = "STRAIGHT"

                if smoothed_track == LEFT:
                    trackWord = "LEFT"
                elif smoothed_track == RIGHT:
                    trackWord = "RIGHT"
                elif smoothed_track == STRAIGHT:
                    trackWord = "STRAIGHT"

                # apply ram to make smooth acceleration
                left = ramp(left, left_prev, ACCELERATION)
                right = ramp(right, right_prev, ACCELERATION)
                
                print("Steer: "+ steerWord, "\t Track: ", trackWord)
                print(left, right)
                ppi.set_velocity(left, right)
                
                left_prev, right_prev = left, right
            



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