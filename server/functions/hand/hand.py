# basics
import numpy as np
from matplotlib import pyplot as plt
# computer vision
import cv2
import mediapipe as mp


# initialize model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# initialize visualization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def visualize_hands(mp_results, img):
    img = img.copy()
    landmarks = mp_results.multi_hand_landmarks
    if landmarks:
        for landmark in landmarks:
            mp_drawing.draw_landmarks(
                img, landmark, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        # print(landmarks)
    return img


def main():
    img = cv2.imread('./tests/test_images/test_hand.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_results = hands.process(img)
    plt.imshow(visualize_hands(mp_results, img))
    plt.show()


if __name__ == "__main__": main()