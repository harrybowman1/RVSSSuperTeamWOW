import cv2
import numpy as np

image = cv2.imread("data/3.jpg")
image = cv2.resize(image,(32,32))
image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)


leftCloseSensor = np.average(image[25:29,4:7],(0,1))
rightCloseSensor = np.average(image[25:29,26:29],(0,1))
centerCloseSensor = np.average(image[25:29,15:18],(0,1))
leftFarSensor = np.average(image[10:13,4:7],(0,1))
rightFarSensor = np.average(image[10:13,26:29],(0,1))
centerFarSensor = np.average(image[10:13,15:18],(0,1))
print (centerCloseSensor)

print(centerCloseSensor[1]<50 or centerCloseSensor[2]<150)
print(leftCloseSensor[1]<50 and centerCloseSensor[2]<150)
print(rightCloseSensor[1]<50 and centerCloseSensor[2]<150)

print(rightCloseSensor[0]<80 and rightCloseSensor[0]>45 and rightCloseSensor[1]>70 and rightCloseSensor[2]>70)
print(leftCloseSensor[0]<80 and leftCloseSensor[0]>45 and leftCloseSensor[1]>70 and leftCloseSensor[2]>70)
print(centerCloseSensor[0]<80 and centerCloseSensor[0]>45 and centerCloseSensor[1]>70 and centerCloseSensor[2]>70)



image[25:29,4:7] = [0,0,0]
image[25:29,26:29] = [0,0,0]
image[25:29,15:18] = [0,0,0]
image[10:13,4:7] = [0,0,0]
image[10:13,15:18] = [0,0,0]
image[10:13,26:29] = [0,0,0]

image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
# color = np.uint8([[[75, 107, 67 ]]])
# print(cv2.cvtColor(color,cv2.COLOR_BGR2HSV))



cv2.imshow("image",image)
cv2.waitKey(5000)