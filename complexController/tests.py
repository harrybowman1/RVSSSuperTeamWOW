import cv2
import numpy as np

image = cv2.imread("data/000001.jpg")
image = cv2.resize(image,(32,32))


print(np.linalg.norm(np.average(image[19:22,4:7],(0,1))-[100,100,100]))
image[19:22,4:7] = [0,0,0]
image[19:22,26:29] = [0,0,0]
cv2.imwrite("edited.jpg",image)


cv2.imshow("image",image)
cv2.waitKey(5000)