import cv2
import numpy as np

class Controller:

    def __init__(self):
        self.speed = 30
        self.time = 0
        self.locationStack = ["no idea where i am"]
        self.commStack = ["comms init"]
        self.generalStack = []

    def steer_away_from_green(img: np.ndarray) -> str:
        """
        Markus
        returns a driving command to drive away from the centroid of the green channel of img
        
        Args:
            img: imput image
            
        Returns:
            either 'LEFT' or 'RIGHT'
        """

        assert img is not None, "no image file found"

        moments = cv2.moments(img[:, :, 1])
        x_centroid = moments['m10'] / moments['m00']
        y_centroid = moments['m01'] / moments['m00']

        (height, width, channels) = img.shape
        assert y_centroid <= height
        assert x_centroid <= width
        assert y_centroid >= 0
        assert y_centroid >= 0
        
        return [x_centroid,y_centroid]

    def loop(self,inputImage):

        #helper stuff
        # cv2.imwrite("data/"+str(self.time).zfill(6)+".jpg", inputImage) 
        self.time+=1
        image = cv2.resize(inputImage,(32,32))
        leftMotor = 0
        rightMotor = 0

        #maintain stacks
        if len(self.commStack)>20:
            self.commStack = self.commStack[:20]
        if len(self.locationStack)>20:
            self.locationStack = self.locationStack[:20]
        if len(self.generalStack)>20:
            self.generalStack = self.generalStack[:20]

        #debugs
        if self.time%20 ==0 and len(self.generalStack)>0:
            while len(self.generalStack)>0:
                print(self.generalStack.pop())



        #periodically spit out comms
        if self.time%20 ==0 and len(self.commStack)>0:
            while len(self.commStack)>0:
                print(self.commStack.pop())


        # vague idea of what road looks like. kinda grey
        road = np.array([100,100,100])
        grass = np.array([130,180,75])

        # small squares on left and right side. average and check if its road
        leftRoadSensor = np.linalg.norm(np.average(image[19:22,4:7],(0,1))-road)<50
        rightRoadSensor = np.linalg.norm(np.average(image[19:22,26:29],(0,1))-road)<50
        # check grass
        leftGrassSensor = np.linalg.norm(np.average(image[19:22,4:7],(0,1))-grass)<50
        rightGrassSensor = np.linalg.norm(np.average(image[19:22,26:29],(0,1))-grass)<50

        # basic control
        if "floor it" in self.generalStack:
            leftMotor = 50
            rightMotor = 50
            self.generalStack.remove("floor it")

        if "turn right" in self.generalStack:
            leftMotor = 40
            rightMotor = 20
            self.generalStack.remove("turn right")
        
        if "turn left" in self.generalStack:
            leftMotor = 20
            rightMotor = 40
            self.generalStack.remove("turn left")

        if "dont see road" in self.locationStack:
            leftMotor=20
            rightMotor=-20
            self.locationStack.remove("dont see road")


        # photovorey detection
        if not leftRoadSensor and not rightRoadSensor:
            self.locationStack.append("dont see road")
        else:
            if leftRoadSensor:
                self.generalStack.append("turn left")
            elif not rightRoadSensor:
                self.generalStack.append("turn left")
            else:
                self.generalStack.append("floor it")
            



        return [leftMotor,rightMotor]