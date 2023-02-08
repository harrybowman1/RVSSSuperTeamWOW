import cv2

class Controller:

    def __init__(self):
        self.speed = 30
        self.time = 0

    def loop(self,inputImage):
        self.time+=1
        cv2.imwrite("data/"+str(self.time).zfill(6)+".jpg", inputImage) 
        return [0,0]