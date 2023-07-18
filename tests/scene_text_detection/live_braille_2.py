
# %% [markdown]
# # **0. Setup**

# %% [markdown]
# Imports

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import dearpygui.dearpygui as dpg
import mediapipe as mp
import time

# %% [markdown]
# Setup

# %%
window_size = [1400, 800]
viz_res = tuple([400, 300])

# %% [markdown]
# # **1. Subcomponents**

# %% [markdown]
# Debug

# %%
debug = True
# debug = False
test_name = 'sign'

# %%
if debug: 
    img_raw = cv2.cvtColor(cv2.imread('./test_images/test_{}.jpg'.format(test_name)), cv2.COLOR_BGR2RGB)
else:
    cam = cv2.VideoCapture(0)
    _, img_raw = cam.read()

img = cv2.resize(img_raw, viz_res, interpolation=cv2.INTER_AREA)
# img_rgba = rgba(img)
plt.imshow(img)

# %% [markdown]
# Utils

# %%
def imdpg2(img):
    return img.astype(np.float32)/255

# %% [markdown]
# ## **1A. Scene Text Detection**

# %%
from rapidocr_onnxruntime import RapidOCR
# ocr = RapidOCR()
ocr = RapidOCR(config_path='./custom.yaml')

# %%
# test

result = ocr(img)
result

# %%
# refactoring needed soon -> highlight() and select() functions

def process(img, texts, sf=1):
    img = img.copy()

    # selection
    min_dist = 200
    selection = None
    center = (int(img.shape[1]/2), int(img.shape[0]/2))


    # highlight mask
    weak = np.zeros(shape=img.shape)
    strong = np.zeros(shape=img.shape)

    # iterate through each text
    for i, r in enumerate(texts): 
        # print(r)

        # points
        pt1 = [int(n*sf) for n in r[0][0]]
        pt2 = [int(n*sf) for n in r[0][2]]

        # add highlight
        try: weak = add_mask(weak, pt1, pt2)
        except Exception: print('highlight error')

        # select
        dist = distance([pt1, pt2], center)
        if dist < min_dist: 
            min_dist = dist
            selection = r

    # selected text
    # print('Selected text: ', selection)
    if selection:
        pt1, pt2 = [int(n*sf) for n in selection[0][0]], [int(n*sf) for n in selection[0][2]]
        pt1, pt2 = tuple(pt1), tuple(pt2) # OpenCV's rectangle doesn't like coordinates given in list
        strong = add_mask(strong, pt1, pt2)
        img = cv2.rectangle(img, pt1, pt2, (255,255,255), 15) # outline

    # visualization
    img = cv2.rectangle(img, center, center, (0,50,0), 20) # center
    highlighted = apply_highlight(img, weak, strong)

    return highlighted, selection
    # return None, None

# %%
frame, sel = process(img, result[0])
# pre-pygame-processing
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# frame = np.rot90(np.fliplr(frame))

# %%
plt.imshow(frame)

# %% [markdown]
# # 1B. **Hand Pose**

# %%
# initialize model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# process
mp_results = hands.process(img)

# %%
# initialize visualization
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# %%
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

# plt.imshow(visualize_hands(mp_results, img))

# %% [markdown]
# # 1C. **Hand Gesture**

# %% [markdown]
# Pose based RNN, debouncing, action-based vs. state-based

# %% [markdown]
# state가 쉬우니까, continuous pointing based 느낌으로 먼저 ㄱㄱ.

# %% [markdown]
# # **2. Optimization**

# %%
# a big TODO

# %% [markdown]
# ONNX, quantization, 등

# %% [markdown]
# global - local kalman filter (camera, in-scene object) assisted residuals-based flicker suppression w/ location estimation

# %% [markdown]
# # **3. User Interaction**

# %% [markdown]
# ## **3A. DearPyGUI**

# %%
# setup DearPyGUI
dpg.create_context()
dpg.create_viewport(title='Live Braille 2', width=window_size[0], height=window_size[1])
dpg.setup_dearpygui()
dpg.show_viewport()

# %%
# texture registry
with dpg.texture_registry(show=True): # show=True
    dpg.add_raw_texture(width=img.shape[1], height=img.shape[0], default_value=imdpg2(img), format=dpg.mvFormat_Float_rgb, tag="stream_visualization")

# %% [markdown]
# ## **3B. DPG Windows**

# %%
with dpg.window(label="Visualization", tag="visualization_window"):
    dpg.add_image("stream_visualization")
dpg.set_item_pos("visualization_window", (0, 0))

# %% [markdown]
# ## **3C. DPG Loop**

# %%
# %%prun -s cumulative -q -l 10 -T mainloop_profile.txt -D mainloop_profile.pstat 

c = 0; start_time = time.time()


while dpg.is_dearpygui_running():
    if debug: pass
    else: 
        _, raw = cam.read(); raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        img = cv2.resize(raw, viz_res, interpolation=cv2.INTER_AREA)

    result = ocr(img)
    frame, sel = process(img, result[0])

    dpg.set_value("stream_visualization", imdpg2(frame))

    dpg.render_dearpygui_frame()
    c += 1


end_time = time.time()
time_delta = end_time - start_time
dpg.destroy_context()

print("Frames Per Second:", round(c/time_delta, 2))

# %% [markdown]
# Text to speech
# 
# Visual description
#   - Optimization using ONNX. Someone has probably done it. 

# %% [markdown]
# 

# %%



