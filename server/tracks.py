# basics
import asyncio
import numpy as np
import time
# aiortc
from aiortc import MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer
from av import VideoFrame
from av import AudioFrame
# computer vision
import cv2
# local modules
from functions.object.object import ObjectDetector
from functions.text.text import TextDetector
from functions.viz.viz import highlight, apply_highlight, add_mask, draw_viz
from functions.utils.utils import imresize
from functions.hand.hand import hands, visualize_hands


BUFFER_SIZE = 1

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
        self.OCR = TextDetector()

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
                    # asyncio.sleep(0.1) # wait a bit. 
                    # time.sleep(0.1) # wait a bit. 
                    await asyncio.to_thread(time.sleep, 0.5)
                    edge = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                    result = edge

                # elif self.transform == "highlight":
                #     weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (100, 100), (300, 300))
                #     strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (200, 200), (400, 400))
                #     highlighted = apply_highlight(img, weak, strong)
                #     result = highlighted
                elif self.transform == "highlight":
                    weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (100, 100), (300, 300))
                    strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (200, 200), (400, 400))
                    highlighted = apply_highlight(img, weak, strong)
                    result = highlighted

                elif self.transform == "text":
                    # TODO make pipeline non-blocking (so that image itself updates while OCR attempts to catch up)
                    # texts = self.OCR(img)
                    # result = img
                    texts = await asyncio.to_thread(self.OCR.detect, img)

                    # weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (100, 100), (300, 300))
                    # strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (200, 200), (400, 400))
                    # highlighted = apply_highlight(img, weak, strong)
                    # result = highlighted
                    
                    result = draw_viz(img, texts=texts[0])

                elif self.transform == "object":
                    dets = self.object.detect(img) # detections
                    drawn = self.object.draw(img, *dets)
                    result = drawn

                elif self.transform == "hand":
                    mp_results = hands.process(img)
                    result = visualize_hands(mp_results, img)
                    
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



class AudioOutputTrack(MediaStreamTrack):
    """
    An audio track that is silent except when it periodically plays an audio file.
    """

    kind = "audio"

    def __init__(self, silence_duration, audio_file):
        super().__init__()  # don't forget this!
        self.silence_duration = silence_duration
        self.audio_file = audio_file
        self.player = None
        self.audio_generator = self.audio_frames()

    async def audio_frames(self):
        while True:
            # Play the audio file
            self.player = MediaPlayer(self.audio_file)
            frame = await self.player.audio.recv()
            while frame is not None:
                yield frame
                frame = await self.player.audio.recv()

            # Yield silence for the silence duration
            start_time = time.time()
            while time.time() - start_time < self.silence_duration:
                # Yield a silent audio frame
                frame = AudioFrame(format="dbl", layout="stereo", samples=48000)
                # Fill frame with zeros for silence
                frame.planes[0].update(np.zeros(48000).astype(np.float64).tobytes())
                yield frame

    async def recv(self):
        return next(self.audio_generator)