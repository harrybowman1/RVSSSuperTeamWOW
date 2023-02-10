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


def getAccuracy(DL, net, dataType = "Val"):
    #Maybe add in net.eval()
    resultsSteer = []
    resultsFuture = []
    for i,data in enumerate(DL,0):
        im = data["image"]    
        y = data['steering'].numpy()[0]
        fs = data['futureSteer'].numpy()[0]

        #Get GT Labels
        label, _= convertLabel(y)
        labelFs, _ = convertLabel(fs)

        #Get outputs from the network
        output = net(im)
        output = torch.squeeze(output)

        #Find out if steer is correct
        outLabel = torch.argmax(output[:3]).numpy()

        if outLabel == label:
            resultsSteer.append(1)
        else:
            resultsSteer.append(0)

        # Find out if steer future is correct
        outLabelFs = torch.argmax(output[3:]).numpy()
        if outLabelFs == labelFs:
            resultsFuture.append(1)
        else:
            resultsFuture.append(0)
    
    accuracySteer = np.sum(np.array(resultsSteer))/len(resultsSteer) * 100
    accuracyFuture = np.sum(np.array(resultsFuture))/len(resultsFuture) * 100
    print(dataType+" Accuracy\n\nSteer Acc = ",accuracySteer, "%\n FutureSteer =  ",accuracyFuture,"%\n\n")
    return accuracySteer, accuracyFuture, resultsSteer, resultsFuture
    
def showClassDist(DL):
    labelList = []
    labelListFS = []
    for i,data in enumerate(DL,0):
        y = data['steering'].numpy()[0]
        fs = data['futureSteer'].numpy()[0]
        label,_ = convertLabel(y)
        labelFS, _ = convertLabel(fs)
        labelList.append(label)
        labelListFS.append(labelFS)
    labels, counts = np.unique(np.array(labelList),return_counts=True)
    labelsFS, countsFS = np.unique(np.array(labelListFS),return_counts=True)
    plt.figure()
    plt.scatter(labels,counts)
    plt.title("0 = left, right = 2, straight = 1")
    plt.figure()
    plt.scatter(labelsFS,countsFS)
    plt.title("0 = left, right = 2, straight = 1")
    plt.figure()
    plt.plot(labelListFS)
    plt.title("Future Steer over time")
    plt.figure()
    plt.plot(labelList)
    plt.title("Steer over time")
    plt.show() 


    

def convertLabel(y):
    thresh = 0.2
    if y > thresh: #Right
        label= 0
    elif y < -thresh: #Left
        label= 2
    else: #Straight
        label= 1
    labelArray = np.array([[0.0,0.0,0.0]])
    labelArray[0][label] = 1.0
    labelTensor = torch.from_numpy(labelArray)
    return label, labelTensor



def modelPredictSteerClass(img, net):
    "Takes in the image and returns the driving command using a trained network"
    im = np.moveaxis(img,2,0)
    cropIm = img[40:,:,:]
    print("imNp = ", cropIm.shape)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    imgTensor = transform(cropIm)
    print("tensor: ",cropIm.shape)
    #Get the image as a tensor and downsize it
    # imgTensor = torch.from_numpy(im)
    # imgTensor = imgTensor.unsqueeze(0)
    # print(imgTensor)

    inputs = TF.resize(imgTensor, [32,32]).unsqueeze(0)
    print(inputs.shape)
    
    # print(type(inputs))
    inputs = inputs.type(torch.float32)
    # print(inputs.shape)
    output = net(inputs)

    output = torch.squeeze(output)

    #Convert Steer
    outSteer = torch.argmax(output[:3]).numpy()
    # print(outSteer)

    #Convert Track Type
    outTrack = torch.argmax(output[3:]).numpy()

    #HERE DOWN IS REDUNDANT
    # print(outTrack)
    if outSteer==2:
        # print("hookin left")
        action= 'LEFT'
    elif outSteer == 1:
        # print("straight down the straight")
        action= "STRAIGHT"
    elif outSteer == 0:
        # print("fangin right")
        action= "RIGHT"
    else:
        raise Exception("Error no turn decided")


    if outTrack==2:
        print("Track left")
    elif outTrack== 1:
        print("Track Straight")
    elif outTrack == 0:
        print("Track right")
    else:
        raise Exception("Error no turn decided")
    # print("\n\n")

    return outSteer, outTrack

