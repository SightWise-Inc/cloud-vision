# basics
import numpy as np
from matplotlib import pyplot as plt
# computer vision
import cv2
import mediapipe as mp


# initialize model
mp_hands = mp.solutions.hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# initialize visualization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class HandDetector:
    def __init__(self):
        self.mp_hands = mp_hands
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.img_shape = None

    # NOTE keep intuitive names (targets: index, thumb, etc, as well as desired gestures)
    # NOTE will this tolerate no hands? 
    def detect(self, img, select='naive'):
        if self.img_shape is None: self.img_shape = img.shape # hacky, maybe
        hands = self.mp_hands.process(img).multi_hand_landmarks
        if hands:
            if select == 'naive': return hands[0]
            if select == 'all': return hands
            if select == 'main': raise NotImplementedError

    def finger_to_landmark(self, finger):
        if type(finger) == int: ldmk_n = finger 
        elif finger == 'thumb': ldmk_n = 4
        elif finger == 'index': ldmk_n = 8
        elif finger == 'middle': ldmk_n = 12
        elif finger == 'ring': ldmk_n = 16
        elif finger == 'pinky': ldmk_n = 20
        return ldmk_n    

    def position(self, hand, finger='index'):
        if hand:
            ldmk_n = self.finger_to_landmark(finger)
            pos = [int(hand.landmark[ldmk_n].x*self.img_shape[1]), int(hand.landmark[ldmk_n].y*self.img_shape[0])] # de-normalize
            return pos

    def gesture(self, landmarks):
        pass



def visualize_hands(hands, img):
    if type(hands) != list: hands = [hands]
    img = img.copy()
    if hands:
        for hand in hands:
            mp_drawing.draw_landmarks(
                # img, landmark, mp_hands.HAND_CONNECTIONS,
                img, hand, mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return img


def main():
    img = cv2.imread('./tests/test_images/test_hand.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hand = HandDetector()
    hand_ = hand.detect(img)
    
    plt.imshow(hand.visualize_hands(hand_, img))
    plt.show()

    # TODO need a better naming scheme.
    # possibly: objdet & objects, textdet & texts, handdet & hands

if __name__ == "__main__": main()