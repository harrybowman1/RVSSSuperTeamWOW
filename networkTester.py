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
from HarrysLibs import *

#Load the model from file
print("using preloaded model")
#Load the model from file
net = CustomBiggerNet()
net.load_state_dict(torch.load('model'))
net.eval()


#Load in the images in a for loop
imageFolder = "dataDirtyTest/*.jpg"
for j,filename in enumerate(glob(imageFolder)):
    img = cv2.imread(filename)

    #Pass into modelPredictSteerClass
    action = modelPredictSteerClass(img, net)
    if j == 100:
        break

