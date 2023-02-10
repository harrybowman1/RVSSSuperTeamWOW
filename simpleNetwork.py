import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
from glob import glob
from os import path
import matplotlib.pyplot as plt 
import torch

# Recapture the data using the proper method so its always the same length with added Ang in the title
# Sort the data loaded in by the numbers at the start of the filename
#Make a dict of filename to average value of the next n
#Use that in the get item step
#Get the average steer value of the next ~15 points and give that to the label


class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        

        #Get the list of filenames
        self.filenames = (glob(path.join(self.root_folder,"*" + self.img_ext)))   

        #Get the appropriate indexs for each file
        self.index = [int(i.split("/")[1].split("Ang")[0]) for i in self.filenames]

        #Sort the indexs by int value
        self.index.sort()

        #Go through and get a sorted list of the steer values by time
        steerValList = []

        self.filenameSort = []
        for i,ind in enumerate(self.index): #For each index
            globMatch = self.root_folder + "*"+str(ind).zfill(6)+"*"
            for j,filename in enumerate(glob(globMatch)): #For each file name matching the index needed
                if j !=0:
                    raise Exception ("More than one file with the same index of "+ str(ind))
                else:
                    #Get the steer values in a sorted list
                    steerValList.append(np.float32(filename.split("Ang")[-1].split(self.img_ext)[0]))
                    self.filenameSort.append(filename)

        #Aggregate into future steer req
        futureSteerReq = []
        nAgg = [0,50] #Aggregate samples into the future: samples[nAgg[0] : nAgg[1]
        for i,steer in enumerate(steerValList):
            if(nAgg[1] + i < len(steerValList)): #Dont do the last ones
                relevantData = np.array(steerValList[i+nAgg[0]:i+nAgg[1]])
                meanVal = np.mean(relevantData)
                futureSteerReq.append(meanVal)
            # else:
            #     print("last ones")
        steerValList = steerValList[:-nAgg[1]]
        self.filenameSort = self.filenameSort[:-nAgg[1]]
        
        #Join these two lists into a feature list to be indexed
        # n x 2 numpy array where n is index
        self.features = np.append(np.array([steerValList]), np.array([futureSteerReq]), axis=0)

        self.totensor = transforms.ToTensor()



    def __len__(self):        
        return len(self.filenameSort)
    
    def __getitem__(self,idx):

        f = self.filenameSort[idx]        
        img = cv2.imread(f)
        cropIm = img[40:,:,:]
        
        if self.transform == None:
            cropIm = self.totensor(cropIm)
        else:
            cropIm = self.transform(cropIm)   
        



        #Downsize the image
        newIm = TF.resize(cropIm, [32,32])

        input = self.features[:,idx]        # steering = f.split("/")[-1].split(self.img_ext)[0][-3:]
        # print("Input size")
        # print(input.shape)
        # steering = np.float32(steering)   
        # if "-" in f:
        #     steering = steering*-1im.shape
    
        sample = {"image":newIm , "steering":input[0], "futureSteer":input[1],"fname":f}        
        
        return sample

#Network to train
class NetPytorchTutorial(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.fc1 = nn.Linear(8 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CustomBiggerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NetFC(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.flatten(x,1) #Turn into flat obj


def test():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SteerDataSet("data/",".jpg",transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)
    print("made data loader")
    for S in ds_dataloader:
        im = S["image"]    
        y  = S["steering"]
        f = S["fname"]
        fs = S['futureSteer']
        # print(len(f))
        # print(y)
        # print(fs)
        # break
    print("Done")
        


if __name__ == "__main__":
    a = test()

