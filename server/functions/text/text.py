# basics
import matplotlib.pyplot as plt
import numpy as np
# computer vision
import cv2
from rapidocr_onnxruntime import RapidOCR
# local modules


OCR = RapidOCR(config_path='./server/functions/text/custom.yaml')
# OCR = RapidOCR(config_path='./wrongpathobviously')
# OCR = RapidOCR()

def main():
    img = cv2.imread('./tests/demo_images/demodemo2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = OCR(img)
    print(result)

if __name__ == "__main__": main()