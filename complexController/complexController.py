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

        #context stacks
        
        
        #comm stack. occasionally give status updates
        
        if self.time%20 ==0 and len(self.commStack)>0:
            print(self.commStack.pop())

        # vague idea of what road looks like. kinda grey
        road = np.array([100,100,100])

        # small squares on left and right side. average and check if its road
        leftRoadSensor = np.linalg.norm(np.average(image[19:22,4:7],(0,1))-road)<50
        rightRoadSensor = np.linalg.norm(np.average(image[19:22,26:29],(0,1))-road)<50

        # normal full speed ahead
        leftMotor = 50
        rightMotor = 50

        # photovorey control
        if not leftRoadSensor:
            leftMotor=0
            rightMotor=20
            if "comm update" in self.generalStack: self.commStack.append("turning right")

        if not rightRoadSensor:
            rightMotor=0
            leftMotor=20
            if "comm update" in self.generalStack: self.commStack.append("turning right")

        if (not leftRoadSensor) and (not rightRoadSensor):
            leftMotor=-10
            rightMotor=10
            if not ("lost the road" in self.locationStack):
                self.locationStack.append("lost the road")
            if "comm update" in self.generalStack: self.commStack.append("turning right")

        # communicate state
        if "comm update" in self.generalStack:
            self.generalStack.remove("comm update")
        if self.time%5==0:
            self.generalStack.append("comm update")


        return [leftMotor,rightMotor]