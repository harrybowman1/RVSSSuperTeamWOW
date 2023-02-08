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

        leftRoadSensor = np.linalg.norm(image[20,5]-road)<50
        rightRoadSensor = np.linalg.norm(image[20,27]-road)<50

        leftMotor = 20
        rightMotor = 20

        if not leftRoadSensor:
            leftMotor+=10
            rightMotor-=10

        # cv2.imwrite("data/"+str(self.time).zfill(6)+".jpg", inputImage) 
        return [leftMotor,rightMotor]