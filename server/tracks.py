# basics
import asyncio
import numpy as np
# aiortc
from aiortc import MediaStreamTrack
from av import VideoFrame
# computer vision
import cv2
# local modules
from functions.object.object import ObjectDetector
from functions.viz.viz import highlight, apply_highlight, add_mask
from functions.utils.utils import distance, imresize

BUFFER_SIZE = 3

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.buffer = asyncio.Queue(maxsize=BUFFER_SIZE)
        self.loop = asyncio.ensure_future(self.process())
        self.result = None
        self.object = ObjectDetector()

    async def recv(self):
        frame = await self.track.recv()
        if self.buffer.full():
            self.buffer.get_nowait() # discard previous frame
        await self.buffer.put(frame)
        # print("added frame")
        if self.result: return self.result
        else: return frame

    async def process(self):
        # print("PROCESS() INVOKED.") # DEBUG
        while True:
            try: 
                # print("processing") # DEBUG

                frame = await self.buffer.get()
                img = frame.to_ndarray(format="bgr24")

                if self.transform == "cartoon":
                    # prepare color
                    img_color = cv2.pyrDown(cv2.pyrDown(img))
                    for _ in range(6):
                        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                    img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                    # prepare edges
                    img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img_edges = cv2.adaptiveThreshold(cv2.medianBlur(img_edges, 7), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2 ,)
                    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                    # combine color and edges
                    img = cv2.bitwise_and(img_color, img_edges)
                    result = img

                elif self.transform == "edges": # perform edge detection
                    edge = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                    result = edge

                elif self.transform == "long":
                    asyncio.sleep(0.1) # wait a bit. 
                    edge = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                    result = edge

                elif self.transform == "highlight":
                    weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (100, 100), (300, 300))
                    strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (200, 200), (400, 400))
                    highlighted = apply_highlight(img, weak, strong)
                    result = highlighted

                elif self.transform == "text":
                    weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (100, 100), (300, 300))
                    strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (200, 200), (400, 400))
                    highlighted = apply_highlight(img, weak, strong)
                    result = highlighted

                elif self.transform == "object":
                    dets = self.object.detect(img) # detections
                    drawn = self.object.draw(img, *dets)
                    result = drawn
                    
                elif self.transform == "rotate":
                    # rotate image
                    img = frame.to_ndarray(format="bgr24")
                    rows, cols, _ = img.shape
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                    img = cv2.warpAffine(img, M, (cols, rows))

                else:
                    result = img
                    # print('no transform') # DEBUG
                    pass

                # rebuild a VideoFrame, preserving timing information
                new_frame = VideoFrame.from_ndarray(result, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                # return new_frame
                self.result = new_frame

            except Exception as e:
                print("EXCEPTION:", e)