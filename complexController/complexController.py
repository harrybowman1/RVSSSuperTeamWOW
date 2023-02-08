import cv2
import numpy as np

class Controller:

    def __init__(self):
        self.speed = 30
        self.time = 0

    def loop(self,inputImage):
        self.time+=1

        image = cv2.resize(inputImage,(32,32))

        road = np.array([100,100,100])

        leftRoadSensor = np.linalg.norm(np.average(image[19:22,4:7],(0,1))-[100,100,100])
        rightRoadSensor = np.linalg.norm(np.average(image[19:22,26:29],(0,1))-[100,100,100])

        leftMotor = 20
        rightMotor = 20

        if not leftRoadSensor:
            leftMotor+=10
            rightMotor-=20

        if not rightRoadSensor:
            rightMotor+=10
            leftMotor-=20


        # cv2.imwrite("data/"+str(self.time).zfill(6)+".jpg", inputImage) 
        return [leftMotor,rightMotor]