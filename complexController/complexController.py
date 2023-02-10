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
        grass = np.array([80,150,80])
        glare = np.array([255,255,255])

        #maintain stacks. empty general stack
        self.generalStack = []


        #define square sensors
        leftCloseSensor = np.average(image[17:20,4:7],(0,1))
        rightCloseSensor = np.average(image[17:20,26:29],(0,1))
        centerCloseSensor = np.average(image[17:20,15:18],(0,1))
        leftEdgeSensor = np.average(image[15:18,0:4],(0,1))
        rightEdgeSensor = np.average(image[15:18,28:],(0,1))
        centerSensor = np.average(image[15:18,15:18],(0,1))



        #check if its road
        leftRoadSensor = np.linalg.norm(leftCloseSensor-road)<70
        rightRoadSensor = np.linalg.norm(rightCloseSensor-road)<70
        centerRoadSensor = np.linalg.norm(centerCloseSensor-road)<70
        leftEdgeRoad = np.linalg.norm(leftEdgeSensor-road)<70
        rightEdgeRoad = np.linalg.norm(rightEdgeSensor-road)<70
        centerFarRoadSensor = np.linalg.norm(centerSensor-road)<70
        leftGrassSensor = np.linalg.norm(leftCloseSensor-grass)<100
        middleGrassSensor = np.linalg.norm(centerSensor-grass)<50
        centerGrassSensor = np.linalg.norm(centerCloseSensor-grass)<50
        centerGlareSensor = np.linalg.norm(centerCloseSensor-glare)<50

        # print(leftFarRoadSensor,centerFarRoadSensor,rightFarRoadSensor)
        print(leftRoadSensor,centerRoadSensor,rightRoadSensor)
        # print(leftGrassSensor,centerGrassSensor,rightGrassSensor)

        # basic control
        if centerRoadSensor and (not middleGrassSensor):
            leftMotor=40
            rightMotor=40
            if not leftEdgeRoad:
                rightMotor-=10
            if not rightEdgeRoad:
                leftMotor-=10
        elif centerRoadSensor and middleGrassSensor:
            leftMotor=10
            rightMotor=10

        else:
            if leftEdgeRoad:
                leftMotor = -10
                rightMotor = 20
            if rightEdgeRoad:
                leftMotor = 20
                rightMotor = -10
            if centerGlareSensor:
                leftMotor=10
                rightMotor=10

        if centerGrassSensor:
            leftMotor=-10
            rightMotor=-10




        if (leftRoadSensor or centerRoadSensor) and not rightRoadSensor:
            leftMotor = -10
            rightMotor = 20
        
        if (rightRoadSensor or centerRoadSensor) and not leftRoadSensor:
            leftMotor = 20
            rightMotor = -10

        if (not centerRoadSensor and not leftRoadSensor and not rightRoadSensor):
            if self.time%120>45:
                leftMotor = 10
                rightMotor = -10
            else:
                leftMotor = -10
                rightMotor = 10


        # if "turn right" in self.generalStack:
        #     leftMotor = 10
        #     rightMotor = -10


        # are we in right or left turn segment?
        # if not leftRoadSensor:
        #     if not "lost road on left" in self.contextStack:
        #         self.contextStack.append("lost road on left")
        #     self.timer1+=1
        # if leftRoadSensor:
        #     if "lost road on left" in self.contextStack:
        #         self.contextStack.remove("lost road on left")
        #         print(self.timer1)
        #     self.timer1 = 0

        # if self.timer1>5:
        #     self.generalStack.append("turn right")


        #debugs. override wheels for debugging
        debug = False
        if debug:
            leftMotor=0
            rightMotor=0

        
    
            



        return [leftMotor,rightMotor]