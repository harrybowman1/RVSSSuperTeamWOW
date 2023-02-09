import cv2
import numpy as np

class Controller:

    def __init__(self):
        self.speed = 30
        self.time = 0
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
        # cv2.imwrite("data/"+str(self.time).zfill(6)+".jpg", inputImage) 
        self.time+=1
        image = cv2.resize(inputImage,(32,32))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        leftMotor = 0
        rightMotor = 0

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
        leftRoadSensor = leftCloseSensor[1]<50 or leftCloseSensor[2]<80
        rightRoadSensor = rightCloseSensor[1]<50 or rightCloseSensor[2]<80
        centerRoadSensor = centerCloseSensor[1]<50 or centerCloseSensor[2]<80
        leftFarRoadSensor = leftFarSensor[1]<50 or leftFarSensor[2]<100
        rightFarRoadSensor = rightFarSensor[1]<50 or rightFarSensor[2]<100
        centerFarRoadSensor = centerFarSensor[1]<50 or centerFarSensor[2]<100
        # check grass
        leftGrassSensor = leftCloseSensor[0]<80 and leftCloseSensor[0]>45 and leftCloseSensor[1]>70 and leftCloseSensor[2]>70
        rightGrassSensor = rightCloseSensor[0]<80 and rightCloseSensor[0]>45 and rightCloseSensor[1]>70 and rightCloseSensor[2]>70
        centerGrassSensor = centerCloseSensor[0]<80 and centerCloseSensor[0]>45 and centerCloseSensor[1]>70 and centerCloseSensor[2]>70
        leftFarGrassSensor = leftFarSensor[0]<80 and leftFarSensor[0]>45 and leftFarSensor[1]>70 and leftFarSensor[2]>70
        rightFarGrassSensor = rightFarSensor[0]<80 and rightFarSensor[0]>45 and rightFarSensor[1]>70 and rightFarSensor[2]>70
        centerFarGrassSensor = centerFarSensor[0]<80 and centerFarSensor[0]>45 and centerFarSensor[1]>70 and centerFarSensor[2]>70

        # photovorey detection
        if centerRoadSensor:
            leftMotor=40
            rightMotor=40
        
        if rightGrassSensor or rightFarGrassSensor:
            leftMotor=0
            rightMotor=20
        
        if leftGrassSensor or leftFarGrassSensor:
            leftMotor=20
            rightMotor=0
        
        if centerGrassSensor:
            leftMotor=-20
            rightMotor=-20

        if centerGrassSensor or centerFarGrassSensor:
            print("grass")

        



        # if not leftRoadSensor and (not rightRoadSensor):
        #     if centerRoadSensor:
        #         self.generalStack.append("slow forward")
        # else:
        #     if leftRoadSensor and rightRoadSensor and centerRoadSensor:
        #         self.generalStack.append("floor it")
        #     else:
        #         if (leftRoadSensor or rightGrassSensor):
        #             self.generalStack.append("turn left") 
        #         elif (rightRoadSensor or leftGrassSensor):
        #             self.generalStack.append("turn right")

                    

        # # basic control
        # if "turn right" in self.generalStack:
        #     leftMotor = 20
        #     rightMotor = 0
        # elif "turn left" in self.generalStack:
        #     leftMotor = 0
        #     rightMotor = 20
        # elif "slow forward" in self.generalStack:
        #     leftMotor=10
        #     rightMotor=10
        # elif "floor it" in self.generalStack:
        #     leftMotor = 50
        #     rightMotor = 50
        # elif "turn around" in self.generalStack:
        #     leftMotor = 5
        #     rightMotor = -5



            
        
        


        #am i on the track?
        # if ("dont know where i am" in self.contextStack):
        #     if(leftFarGrassSensor or rightFarGrassSensor or centerFarGrassSensor):
        #         self.contextStack.remove("dont know where i am")
        #         self.contextStack.append("grass track in front")
        #     else:
        #         self.generalStack.append("revolve")


        if ("revolve" in self.generalStack):
            leftMotor = -10
            rightMotor = 10



        #debugs. override wheels for debugging
        debug = False
        if debug:
            leftMotor=0
            rightMotor=0

        
    
            



        return [leftMotor,rightMotor]