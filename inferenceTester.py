from simpleNetwork import *
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

#~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
pygame.init()
pygame.display.set_mode((300,300)) #size of pop-up window
pygame.key.set_repeat(100) #holding a key sends continuous KEYDOWN events. Input argument is milli-seconds delay between events and controls the sensitivity
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("initialise camera")
camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')

#In a forever loop, get new image, and send commands to the motors
