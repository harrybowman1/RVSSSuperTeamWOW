import cv2
import numpy as np

class Controller:

    def __init__(self):
        self.speed = 30
        self.time = 0

    def loop(self,inputImage):

        #helper stuff
        self.time+=1
        image = cv2.resize(inputImage,(32,32))

        #context stack
        locationStack = ["no idea where i am"]

        # vague idea of what road looks like. kinda grey
        road = np.array([100,100,100])

        # small squares on left and right side. average and check if its road
        leftRoadSensor = np.linalg.norm(np.average(image[19:22,4:7],(0,1))-road)<50
        rightRoadSensor = np.linalg.norm(np.average(image[19:22,26:29],(0,1))-road)<50

        # normal full speed ahead
        leftMotor = 20
        rightMotor = 20

        # photovorey control
        if not leftRoadSensor:
            leftMotor+=10
            rightMotor-=20

        if not rightRoadSensor:
            rightMotor+=10
            leftMotor-=20

        if (not leftRoadSensor) and (not rightRoadSensor):
            leftMotor=-10
            rightMotor=10

        # cv2.imwrite("data/"+str(self.time).zfill(6)+".jpg", inputImage) 
        return [leftMotor,rightMotor]