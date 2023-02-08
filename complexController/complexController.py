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

        leftRoadSensor = np.linalg.norm(image[20,5]-road)<100

        print(leftRoadSensor)

    

        # cv2.imwrite("data/"+str(self.time).zfill(6)+".jpg", inputImage) 
        return [0,0]