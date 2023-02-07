"""compute the mean of all green pixels and drive away from it
"""

#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import penguinPi as ppi

INNER_WHEEL = 25
OUTER_WHEEL = 35


def steer_away_from_green(img: np.ndarray) -> str:
    """returns a driving command to drive away from the centroid of the green channel of img
    
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
    
    if x_centroid < width/2:
        return "RIGHT"

    return "LEFT"


def drive():
    camera = ppi.VideoStreamWidget('http://localhost:8080/camera/get')

    while True:
        image = camera.frame
        print(type(image))
        command = steer_away_from_green(image)
        if command == "LEFT":
            ppi.set_velocity(INNER_WHEEL, OUTER_WHEEL) 
        elif command == "RIGHT":
            ppi.set_velocity(OUTER_WHEEL, INNER_WHEEL) 
        else:
            assert False


if __name__ == "__main__":
    img = cv2.imread('./on_robot/collect_data/data/000146-0.50.jpg', cv2.IMREAD_COLOR)
    command = away_from_green(img)
    print(command)
