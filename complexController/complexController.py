import cv2
import numpy as np

class Controller:

    def __init__(self):
        self.speed = 30
        self.time = 0
        self.generalStack = []
        self.contextStack = []

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
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        leftMotor = 0
        rightMotor = 0

        #maintain stacks. empty general stack
        self.generalStack = []


        #periodically spit out comms
        if self.time%20 ==0 and len(self.commStack)>0:
            while len(self.commStack)>0:
                print(self.commStack.pop())



        #define square sensors
        leftCloseSensor = np.average(image[25:29,4:7],(0,1))
        rightCloseSensor = np.average(image[25:29,26:29],(0,1))
        centerCloseSensor = np.average(image[25:29,15:18],(0,1))
        leftFarSensor = np.average(image[10:13,4:7],(0,1))
        rightFarSensor = np.average(image[10:13,26:29],(0,1))
        centerFarSensor = np.average(image[10:13,15:18],(0,1))

        # small squares on left and right side. average and check if its road
        leftRoadSensor = leftCloseSensor[1]<50 or leftCloseSensor[2]<150
        rightRoadSensor = rightCloseSensor[1]<50 or rightCloseSensor[2]<150
        centerRoadSensor = centerCloseSensor[1]<50 or centerCloseSensor[2]<150
        # check grass
        leftGrassSensor = False
        rightGrassSensor = False

        # photovorey detection
        if not leftRoadSensor and not rightRoadSensor:
            self.generalStack.append("turn around")
        else:
            if leftRoadSensor and rightRoadSensor and centerRoadSensor:
                self.generalStack.append("floor it")
            else:
                if leftRoadSensor or not rightRoadSensor or rightGrassSensor:
                    self.generalStack.append("turn left")
                elif rightRoadSensor or not leftRoadSensor or leftGrassSensor:
                    self.generalStack.append("turn right")

        # basic control
        if "floor it" in self.generalStack:
            leftMotor = 50
            rightMotor = 50
            self.generalStack.remove("floor it")
            
        elif "turn right" in self.generalStack:
            leftMotor = 40
            rightMotor = 20
            self.generalStack.remove("turn right")
        
        elif "turn left" in self.generalStack:
            leftMotor = 20
            rightMotor = 40
            self.generalStack.remove("turn left")

        elif "turn around" in self.generalStack:
            leftMotor=0
            rightMotor=0
            self.generalStack.remove("turn around")


        
    
            



        return [leftMotor,rightMotor]