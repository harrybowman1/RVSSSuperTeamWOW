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

ds = SteerDataSet("data1/",".jpg",transform)

print("The dataset contains %d images " % len(ds))

ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)

#Import the network definedd in simple network
net = NetPytorchTutorial()

#define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(ds_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        im = data["image"]    
        y  = data["steering"]
        if y > 0: #left
           label= torch.from_numpy(np.array([[1.0,0.0,0.0]]))
        elif y < 0: #right
            label= torch.from_numpy(np.array([[0.0,1.0,0.0]]))
        else: #Straight
            label= torch.from_numpy(np.array([[0.0,0.0,1.0]]))
        

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
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(),"model")