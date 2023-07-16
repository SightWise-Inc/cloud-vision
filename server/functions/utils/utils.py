import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma


# Distance to rectangle
def distance(rect, point):
    dx = max(min(rect[0][0], rect[1][0]) - point[0], 0, point[0] - max(rect[0][0], rect[1][0]))
    dy = max(min(rect[0][1], rect[1][1]) - point[1], 0, point[1] - max(rect[0][1], rect[1][1]))
    # dx = max(rect.bottom_left.x - point[0], 0, point[0] - (rect.bottom_left.x+rect.width))
    # dy = max(rect.bottom_left.y - point[1], 0, point[1] - (rect.bottom_left.y+rect.height))
    return math.sqrt(dx*dx + dy*dy)


# Resize image
def imresize(img):
    if img.shape[1] < img.shape[0]: # portrait or landscape?
        coef = 300/img.shape[1]
    else:
        coef = 300/img.shape[0]
    # if coef < 1: return img # make sure not to make it bigger

    dim = (int(img.shape[1]*coef), int(img.shape[0]*coef))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA), 1/coef


def main():
    # distance
    rectangle = [(-1,1),(1,-1)]
    point_1 = (0,0) # should be true
    point_2 = (1,1) # should be true ideally (outline inclusive)
    point_3 = (1.1,1.1) # should be false

    print(distance(rectangle, point_1))
    print(distance(rectangle, point_2))
    print(distance(rectangle, point_3))


if __name__ == "__main__": main()