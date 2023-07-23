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
from functions.utils.utils import imresize, select
from functions.hand.hand import HandDetector


BUFFER_SIZE = 1

class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform, loop):
        super().__init__()
        self.track = track
        self.transform = transform
        self.loop = loop
        # pipeline
        self.buffer = asyncio.Queue(maxsize=BUFFER_SIZE)
        self.loop = asyncio.ensure_future(self.process())
        self.result = None
        # computer vision models
        self.object = ObjectDetector()
        self.text = TextDetector()
        self.hand = HandDetector()

    async def recv(self):
        frame = await self.track.recv()
        if self.buffer.full():
            self.buffer.get_nowait() # discard previous frame
        await self.buffer.put(frame)
        # print("added frame")
        if self.result: return self.result
        else: return frame

    async def process(self):
        while True:
            try: 
                frame = await self.buffer.get()
                img = frame.to_ndarray(format="bgr24")

                if self.transform == "edges": # perform edge detection
                    edge = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                    result = edge

                elif self.transform == "long":
                    await asyncio.to_thread(time.sleep, 0.5)
                    edge = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
                    result = edge

                elif self.transform == "highlight":
                    weak = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (100, 100), (300, 300))
                    strong = add_mask(np.zeros(shape=img.shape, dtype=np.uint8), (200, 200), (400, 400))
                    highlighted = apply_highlight(img, weak, strong)
                    result = highlighted

                # TODO make pipeline non-blocking (so that image itself updates while OCR attempts to catch up)
                elif self.transform == "text":
                    texts = await asyncio.to_thread(self.text.detect, img)
                    drawn = draw_viz(img, texts=texts)
                    result = drawn

                elif self.transform == "object":
                    objects = await asyncio.to_thread(self.object.detect, img)
                    drawn = draw_viz(img, objects=objects)
                    result = drawn

                elif self.transform == "hand":
                    # hands = self.hand.detect(img, select='naive')
                    hand = await asyncio.to_thread(self.hand.detect, img, select='naive')
                    result = draw_viz(img, hands=hand)

                # TODO selection temporal substance
                # TODO parse information into a consistent data type (possibly dict. verbose is good.)
                    # NOTE columnwise selection is a req, so don't make everything a dict/json though
                        # NOTE option 1) keep a well-documented, consistent ndarray
                        # NOTE option 2) group everything into properties then use idx to refer to a certain instance
                        # NOTE option 3) can just make some helpers to intuitively refer to stuff too. that seems smart.
                # NOTE 이거 앱으로 어떻게 만들지 진짜? ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
                # TODO pool models into a shared asyncio pool
                elif self.transform == "mvp":
                    objects = None # NOTE temp # objects = await asyncio.to_thread(self.object.detect, img)
                    texts = await asyncio.to_thread(self.text.detect, img)
                    hand = await asyncio.to_thread(self.hand.detect, img, select='naive')
                    target = self.hand.position(hand)
                    selection = select(objects, texts, target)
                    drawn = draw_viz(img, objects=objects, texts=texts, hands=hand, selection=selection, cursor=target)
                    result = drawn
                    
                elif self.transform == "rotate":
                    img = frame.to_ndarray(format="bgr24")
                    rows, cols, _ = img.shape
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                    img = cv2.warpAffine(img, M, (cols, rows))

                else:
                    result = img
                    pass

            except Exception as e:
                result = img
                print("EXCEPTION:", e)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(result, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            # return new_frame
            self.result = new_frame



from gtts import gTTS
import io

class AudioOutputTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self):
        super().__init__()
        self.player = None
        self.audio_generator = None
        self.busy = False
        self.last_spoken = None

    async def start(self, text):
        if not self.busy:
            self.busy = True
            self.audio_generator = self.audio_frames(text)
            self.last_spoken = text

    async def audio_frames(self, text):
        # Generate audio file from text using gTTS
        if text:
            tts = gTTS(text=text)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)

             # Play audio file using MediaPlayer
            self.player = MediaPlayer(fp)
            self.player.play()

        # Yield audio frames until audio file finishes playing
        while True:
            frame = await self.player.get_frame()
            if frame is None: 
                self.busy = False
                break
            yield frame

            # # Yield silence for the silence duration
            # start_time = time.time()
            # while time.time() - start_time < self.silence_duration:
            #     # Yield a silent audio frame
            #     frame = AudioFrame(format="dbl", layout="stereo", samples=48000)
            #     # Fill frame with zeros for silence
            #     frame.planes[0].update(np.zeros(48000).astype(np.float64).tobytes())
            #     yield frame

    async def recv(self):
        return next(self.audio_generator)