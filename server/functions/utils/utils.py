import cv2
import matplotlib.pyplot as plt
import numpy as np


# Resize image
def imresize(img):
    if img.shape[1] < img.shape[0]: # portrait or landscape?
        coef = 300/img.shape[1]
    else:
        coef = 300/img.shape[0]
    # if coef < 1: return img # make sure not to make it bigger

    dim = (int(img.shape[1]*coef), int(img.shape[0]*coef))
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA), 1/coef


def select(img, dets=None, texts=None, hand=None):

    # iterate through each text/objects
    for i, r in enumerate(texts): 
        # print(r)

        # points
        pt1 = [int(n) for n in r[0][0]]
        pt2 = [int(n) for n in r[0][2]]

        # add highlight
        try: weak = add_mask(weak, pt1, pt2)
        except Exception: print('highlight error')

        # select
        dist = distance([pt1, pt2], center)
        if dist < min_dist: 
            min_dist = dist
            selection = r


def main():
    pass

if __name__ == "__main__": main()