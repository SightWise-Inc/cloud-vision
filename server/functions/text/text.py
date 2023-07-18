# basics
import matplotlib.pyplot as plt
import numpy as np
# computer vision
import cv2
# from rapidocr_onnxruntime import RapidOCR
from paddleocr import PaddleOCR, draw_ocr
# local modules
# from functions.viz.viz import highlight, apply_highlight, add_mask, draw_viz # DEBUG # NOTE doesn't work



# OCR = RapidOCR(config_path='./server/functions/text/custom.yaml')



class TextDetector():
    def __init__(self):
        self.ocr = PaddleOCR(lang="en", show_log = False)

    def detect(self, img):
        return self.ocr.ocr(img, cls=False)



def main():
    img = cv2.imread('./tests/test_images/text/demodemo2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ocr = TextDetector()
    texts = ocr.detect(img)
    print(texts)

    # draw_viz(img, texts=result) # NOTE seems like it's time to figure out how to do sibling imports. or separate tests to a separate folder (personally I don't want this)

    # NOTE temporary way to test viz compatibility. 
    # if texts:
    #     for i, text in enumerate(texts[0]): 
    #         # print(text)
    #         pt1 = [int(n) for n in text[0][0]]
    #         pt2 = [int(n) for n in text[0][2]]

    #         print(pt1)
    #         print(pt2)


if __name__ == "__main__": main()