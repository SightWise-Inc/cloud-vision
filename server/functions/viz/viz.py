import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# Highlight image
def highlight(img, pt1, pt2):
    img = img.copy()

    alpha = 2.2 # Simple contrast control
    beta = 30    # Simple brightness control

    hl = img[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    hl = np.clip(alpha*hl + beta, 0, 255) # option 1: numpy
    # hl = cv2.convertScaleAbs(hl, alpha=alpha, beta=beta) # option 2: cv2
    img[pt1[1]:pt2[1],pt1[0]:pt2[0]] = hl
    print('dtype of img:', img.dtype, '', 'dtype of hl:', hl.dtype)

    return img
# modified from https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html


# add mask for selective image manipulation
def add_mask(mask, pt1, pt2):
    mask[pt1[1]:pt2[1],pt1[0]:pt2[0]] = 1
    return mask


# add mask for selective image manipulation
def apply_highlight(img, weak_mask, strong_mask):
    # weak -= np.logical_and(weak, strong) # remove overlap # unnecessary because strong overwrites weak

    alpha = 2.4 # Simple contrast control
    beta = 30    # Simple brightness control

    weak = np.clip(img * alpha/1.7 + beta/2, 0, 255).astype(np.uint8)
    strong = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)

    highlighted = np.where(weak_mask, weak, img)
    highlighted = np.where(strong_mask, strong, highlighted)

    return highlighted


def main():
    # masking
    a = np.zeros(shape=[4,4])
    a = add_mask(a, (2,2), (3,3))
    print(a)

    folder = './tests/demo_images/'
    filename = 'demodemo2.jpg'
    print(os.path.exists(folder+filename))
    img = cv2.imread(folder+filename)
    
    # highlighted = highlight(img, (926, 1478), (2021, 1756))
    # print(highlighted.dtype)
    # plt.imshow(highlighted)
    weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (926, 1478), (2021, 1756))
    strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (1026, 1578), (2121, 1856))
    plt.imshow(apply_highlight(img, weak, strong)) # test
    plt.show()


if __name__ == "__main__": main()