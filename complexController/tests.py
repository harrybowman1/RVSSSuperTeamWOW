import cv2
import numpy as np

image = cv2.imread("data/000001.jpg")
image = cv2.resize(image,(32,32))


print(np.linalg.norm(image[20,5]-image[8,1]))
cv2.imwrite("edited.jpg",image)


cv2.imshow("image",image)
cv2.waitKey(5000)