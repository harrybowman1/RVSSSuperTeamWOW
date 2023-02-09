import cv2
import numpy as np

image = cv2.imread("data/1.jpg")
image = cv2.resize(image,(32,32))
image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)


leftCloseSensor = np.average(image[25:29,4:7],(0,1))
rightCloseSensor = np.average(image[25:29,26:29],(0,1))
centerCloseSensor = np.average(image[25:29,15:18],(0,1))
leftFarSensor = np.average(image[10:13,4:7],(0,1))
rightFarSensor = np.average(image[10:13,26:29],(0,1))
centerFarSensor = np.average(image[10:13,15:18],(0,1))

print(centerCloseSensor[1]<50 or centerCloseSensor[2]<150)
print(leftCloseSensor[1]<50 and centerCloseSensor[2]<150)
print(rightCloseSensor[1]<50 and centerCloseSensor[2]<150)

image[25:29,4:7] = [0,0,0]
image[25:29,26:29] = [0,0,0]
image[25:29,15:18] = [0,0,0]
image[10:13,4:7] = [0,0,0]
image[10:13,15:18] = [0,0,0]
image[10:13,26:29] = [0,0,0]

image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
color = np.uint8([[[100,100,100 ]]])
print(cv2.cvtColor(color,cv2.COLOR_BGR2HSV))



cv2.imshow("image",image)
cv2.waitKey(5000)