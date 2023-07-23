import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from functions.object.object import ObjectDetector, xywh2xyxy
from functions.text.text import TextDetector
from functions.hand.hand import HandDetector
from functions.viz.viz import highlight, apply_highlight, add_mask, draw_viz


# MIN_DIST = 200
MIN_DIST = 1000


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


# NOTE still such an imperfect implementation. it's possible that the selections flicker. what are we gonna do about it?
select_history = None
def select(objects, texts, target=None):
    global select_history

    # handle temporality

    if target is None: return None

    closest = None
    min_dist = MIN_DIST
    if objects: 
        if objects['boxes'].size > 0:

            # OPTION 1
            # for idx, object in enumerate(objects['boxes']):
            #     print(object) # DEBUG
            #     # extract points
            #     pt1 = [int(n) for n in object[0]]
            #     pt2 = [int(n) for n in object[2]]
            #     # select

            # OPTION 2 (don't fully understand this one)
            boxes = objects['boxes']
            scores = objects['scores']
            class_ids = objects['class_ids']
            indices = objects['indices']
            for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):

                print(label, bbox) # DEBUG
                pt1 = [bbox[0], bbox[1]]
                pt2 = [bbox[2], bbox[3]]

                dist = distance([pt1, pt2], target)
                print('Object', label, ':', dist) # DEBUG
                if dist < min_dist: 
                    closest = { 'box': [pt1, pt2], 'name': label }
                    min_dist = dist

    if texts:
        for idx, text in enumerate(texts): 
            # print(text) # DEBUG
            # extract points
            pt1 = [int(n) for n in text[0][0]]
            pt2 = [int(n) for n in text[0][2]]
            # select
            dist = distance([pt1, pt2], target)
            print('Text', text[1], ':', dist) # DEBUG
            if dist < min_dist: 
                # closest = text
                closest = { 'box': [pt1, pt2], 'name': text[1] } # hopefully this works
                min_dist = dist

    return closest


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


def test_distance():
    rectangle = [(-1,1),(1,-1)]
    point_1 = (0,0) # should be true
    point_2 = (1,1) # should be true ideally (outline inclusive)
    point_3 = (1.1,1.1) # should be false

    print(distance(rectangle, point_1))
    print(distance(rectangle, point_2))
    print(distance(rectangle, point_3))


def test_masking(img):
    a = np.zeros(shape=[4,4])
    a = add_mask(a, (2,2), (3,3))
    print(a)
    
    weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (926, 1478), (2021, 1756))
    strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (1026, 1578), (2121, 1856))
    plt.imshow(apply_highlight(img, weak, strong)) # test
    plt.show()


def test_selection(img):
    object = ObjectDetector()
    objects = object.detect(img)

    text = TextDetector()
    texts = text.detect(img)

    hand = HandDetector()
    hand_ = hand.detect(img, select='naive')
    target = hand.position(hand_)

    selection = select(objects, texts, target)
    if selection: print('Selection:', selection)

    drawn = draw_viz(img, objects=objects, texts=texts, hands=hand_, selection=selection, cursor=target)

    plt.imshow(drawn)
    plt.show()


def main():
    # folder = './tests/test_images/text/'
    # filename = 'demodemo2.jpg'
    folder = './tests/test_images/Selection/Savanna/'
    filename = 'Selection-Savanna-BigElephant.jpg'
    # folder = './tests/test_images/'
    # filename = 'test_desk.jpg'
    img = cv2.imread(folder+filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # test_distance()
    # test_masking(img)
    test_selection(img)


if __name__ == "__main__": main()