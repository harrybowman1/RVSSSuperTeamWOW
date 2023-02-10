import cv2
import numpy as np

class Controller:

    def __init__(self):
        self.time = 0
        self.timer1 = 0
        self.generalStack = []
        self.contextStack = ["dont know where i am"]

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
        self.time+=1
        image = cv2.resize(inputImage,(32,32))
        leftMotor = 0
        rightMotor = 0
        road = np.array([100,100,100])

        #maintain stacks. empty general stack
        self.generalStack = []


        #define square sensors
        leftCloseSensor = np.average(image[25:29,4:7],(0,1))
        rightCloseSensor = np.average(image[25:29,26:29],(0,1))
        centerCloseSensor = np.average(image[25:29,15:18],(0,1))
        leftFarSensor = np.average(image[10:13,4:7],(0,1))
        rightFarSensor = np.average(image[10:13,26:29],(0,1))
        centerFarSensor = np.average(image[10:13,15:18],(0,1))



        #check if its road
        leftRoadSensor = np.linalg.norm(leftCloseSensor-road)<50
        rightRoadSensor = np.linalg.norm(rightCloseSensor-road)<50
        centerRoadSensor = np.linalg.norm(centerCloseSensor-road)<50
        leftFarRoadSensor = np.linalg.norm(leftFarSensor-road)<50
        rightFarRoadSensor = np.linalg.norm(rightFarSensor-road)<50
        centerFarRoadSensor = np.linalg.norm(centerFarSensor-road)<50

        # print(leftFarRoadSensor,centerFarRoadSensor,rightFarRoadSensor)
        # print(leftRoadSensor,centerRoadSensor,rightRoadSensor)

        # basic control
        if centerRoadSensor:
            leftMotor=20
            rightMotor=20

        if (leftRoadSensor or centerRoadSensor) and not rightRoadSensor:
            leftMotor = -10
            rightMotor = 20
        
        if (rightRoadSensor or centerRoadSensor) and not leftRoadSensor:
            leftMotor = 20
            rightMotor = -10

        # if "turn right" in self.generalStack:
        #     leftMotor = 10
        #     rightMotor = -10


        # are we in right or left turn segment?
        if not leftRoadSensor:
            if not "lost road on left" in self.contextStack:
                self.contextStack.append("lost road on left")
            self.timer1+=1
        if leftRoadSensor:
            if "lost road on left" in self.contextStack:
                self.contextStack.remove("lost road on left")
                print(self.timer1)
            self.timer1 = 0

        if self.timer1>5:
            self.generalStack.append("turn right")


        #debugs. override wheels for debugging
        debug = False
        if debug:
            leftMotor=0
            rightMotor=0

        
    
            



        return [leftMotor,rightMotor]