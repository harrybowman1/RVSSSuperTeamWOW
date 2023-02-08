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


#Import the dataset
transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ds = SteerDataSet("dataNew/",".jpg",transform)
dsVal = SteerDataSet("dataNewVal/",".jpg", transform)

print("The dataset contains %d images " % len(ds))

ds_DL = DataLoader(ds,batch_size=1,shuffle=True)
dsVal_DL = DataLoader(dsVal,batch_size=1,shuffle=True)

#Import the network definedd in simple network
net = NetPytorchTutorial()

#define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def getAccuracy(DL, model):
    #Maybe add in net.eval()
    results = []
    for i,data in enumerate(DL,0):
        im = data["image"]    
        y  = data["steering"]
        if y > 0: #left
           label= 0
        elif y < 0: #right
            label= 2
        else: #Straight
            label= 1

        output = net(im)
        outLabel = torch.argmax(output,1)
        if outLabel == label:
            results.append(1)
        else:
            results.append(0)
    
    accuracy = np.sum(np.array(results))/len(results) * 100
    print("Validation Accuracy = : ",accuracy, "%\n\n\n")
    return accuracy, results


trainNew = False
if trainNew:
    valAccList = []
    lossList = []
    bestValAcc = 0

    for epoch in range(10):  # loop over the dataset multiple times
        print("New epoch")
        running_loss = 0.0
        for i, data in enumerate(ds_DL, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            im = data["image"]    
            y  = data["steering"]
            if y > 0.2: #left
                label= torch.from_numpy(np.array([[1.0,0.0,0.0]]))
            elif y < 0.2: #right
                label= torch.from_numpy(np.array([[0.0,0.0,1.0]]))
            else: #Straight
                label= torch.from_numpy(np.array([[0.0,1.0,0.0]]))
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(im)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                lossList.append(running_loss / 100)
                running_loss = 0.0
        accuracy, _= getAccuracy(dsVal_DL, net)
        valAccList.append(accuracy)
        if accuracy > bestValAcc:
            print('New PB, go you!\n\n\n')
            bestValAcc = accuracy
            torch.save(net.state_dict(),"model")
        print("end")


    plt.figure(0)
    plt.plot(valAccList)
    plt.title("Val Acc")
    plt.figure(1)
    plt.plot(lossList)
    plt.title("loss")
    plt.show()
                

                


else:
    print("using preloaded model")
    #Load the model from file
    net = NetPytorchTutorial()
    net.load_state_dict(torch.load('model'))
    net.eval()
    print('val data')
    results = getAccuracy(dsVal_DL, net)
    print('training data')
    results = getAccuracy(ds_DL, net)
    print('done')
