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
from HarrysLibs import *
 
plt.close('all')

#Import the dataset
transform = transforms.Compose(
[transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

ds = SteerDataSet("data/",".jpg",transform)
dsTrain, dsVal = torch.utils.data.random_split(ds, [0.8, 0.2])

dsTest = SteerDataSet("dataValidate/",".jpg", transform)

dsNewTest = SteerDataSet("dataDirtyTest/", ".jpg",transform)

print("The dataset contains %d images " % len(ds))

ds_DL = DataLoader(dsTrain,batch_size=1,shuffle=True)
dsVal_DL = DataLoader(dsVal,batch_size=1,shuffle=True)
dsAll_DL = DataLoader(ds,batch_size=1,shuffle=False)
dsTest_DL = DataLoader(dsTest,batch_size=1,shuffle=False)
dsTestDirty_DL = DataLoader(dsNewTest,batch_size=1,shuffle=False)

#Import the network definedd in simple network
net = CustomBiggerNet()

#define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#Show the class dist
# showClassDist(dsAll_DL)



#Do the training
trainNew = True
if trainNew:
    valAccListSteer = []
    valAccListFuture = []
    testAccListSteer = []
    testAccListFuture = []
    lossList = []
    bestValAcc = 0

    for epoch in range(20):  # loop over the dataset multiple times
        print("New epoch")
        running_loss = 0.0
        for i, data in enumerate(ds_DL, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            im = data["image"]    
            y = data['steering'].numpy()[0]
            fs = data['futureSteer'].numpy()[0]
            _, label = convertLabel(y)
            _, labelFS = convertLabel(fs)

            labelComb = torch.cat((label, labelFS)).flatten()
            labelComb = torch.unsqueeze(labelComb, dim=0)
            

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(im)
            loss = criterion(outputs, labelComb)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                lossList.append(running_loss / 100)
                running_loss = 0.0
        # Test on the validation
        accuracySteer, accuracyFuture, _,_= getAccuracy(dsTestDirty_DL, net)
        valAccListSteer.append(accuracySteer)
        valAccListFuture.append(accuracyFuture)

        if accuracySteer > bestValAcc:
            print('New PB, go you!\n\n\n')
            bestValAcc = accuracySteer
            torch.save(net.state_dict(),"model")
            print("Saved new model")

        #Perform test on the test dataset
        accuracySteer, accuracyFuture, _,_= getAccuracy(dsTest_DL, net, dataType="Test")
        testAccListSteer.append(accuracySteer)
        testAccListFuture.append(accuracyFuture)
        print("end")

    plt.figure()
    plt.plot(valAccListSteer)
    plt.title("Val Acc Steer")
    plt.figure()
    plt.plot(valAccListFuture)
    plt.title("Val Acc Future")

    plt.figure()
    plt.plot(testAccListSteer)
    plt.title("Val Acc Steer")
    plt.figure()
    plt.plot(testAccListFuture)
    plt.title("Val Acc Future")

    plt.figure()
    plt.plot(lossList)
    plt.title("loss")
    plt.show()

else:
    print("using preloaded model")
    #Load the model from file
    net = CustomBiggerNet()
    net.load_state_dict(torch.load('model'))
    net.eval()
    print('dirtyData data')
    results = getAccuracy(dsTestDirty_DL, net)
    print('training data')
    # results = getAccuracy(ds_DL, net)
    print('done')


labelNew = False
if labelNew:
    labelList = ['left', 'straight', 'right']
    dsNew = SteerDataSet("dataTest/",".jpg",transform)
    newDL = DataLoader(dsNew,batch_size=1,shuffle=False)
    for i, data in enumerate(newDL, 0):
        im = data["image"]
        y = data['steering'].numpy()[0]
        print(y)
        GTlabel,_ = convertLabel(y)
        print(GTlabel)
        output = net(im)
        outLabel = torch.argmax(output,1)

        #Get im to display
        a = im.numpy()[0]
        rgbIm = np.moveaxis(a,0,-1)
        plt.imshow(rgbIm)
        savePath = "dataTest/"+str(i) + " Label predicted " + str(labelList[outLabel]) + "Label GT "+str(labelList[GTlabel])
        plt.savefig(savePath)


