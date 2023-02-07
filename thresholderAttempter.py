import cv2
import numpy as np 
import matplotlib.pyplot as plt 
import machinevisiontoolbox as mvt 
import glob


print('wassssaaahhhp')
plt.close('all')
i=0
for name in glob.glob('data/*'):
    i+=1
    image = cv2.imread(name)
    imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    desShape = image.shape[0:2]
    pixels = np.float32(imHSV.reshape(-1,3))
    green = image.copy()
    green[:,:,0] = 0
    green[:,:,2] = 0

    #Get RGB info - dont use blue
    colRange = 60
    bc = 130
    gc = 205
    red = np.array([360-colRange, colRange]) / 2
    green = np.array([bc-colRange, bc+colRange])/2
    blue = np.array([gc-colRange, gc+colRange])/2
    i_pixels = np.array([i if i[1]> 20 else np.array([0.1,0.1,0.1]) for i in pixels]) #Threshold intensities for the pixels
    r_pixels = np.array([1 if ((i < red[1] or i > red[0]) and i != 0.1) else 0 for i in i_pixels[:,0]])
    b_pixels = np.array([1 if ((i > blue[0] and i < blue[1]) and i != 0.1) else 0 for i in i_pixels[:,0]])
    g_pixels = np.array([1 if ((i > green[0] and i < green[1]) and i != 0.1 )else 0 for i in i_pixels[:,0]])

    # cv2.imshow('mask')
    title = str(i) + 'green'
    # cv2.imshow(title,green)
    cv2.imshow(str(i),image)
    # cv2.imshow('1',imHSV)
    title = title+'binarey'
    plt.figure()
    plt.imshow(g_pixels.reshape(desShape))
    # cv2.destroyAllWindows()
    print(name)

    if i > 3:
        break


plt.show()