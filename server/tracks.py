import asyncio
import cv2
from aiortc import MediaStreamTrack
from av import VideoFrame

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

                if self.transform == "cartoon":
                    img = frame.to_ndarray(format="bgr24")

                    # prepare color
                    img_color = cv2.pyrDown(cv2.pyrDown(img))
                    for _ in range(6):
                        img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                    img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                    # prepare edges
                    img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img_edges = cv2.adaptiveThreshold(
                        cv2.medianBlur(img_edges, 7),
                        255,
                        cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY,
                        9,
                        2,
                    )
                    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                    # combine color and edges
                    img = cv2.bitwise_and(img_color, img_edges)

                    # rebuild a VideoFrame, preserving timing information
                    new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    # return new_frame
                    self.result = new_frame
                elif self.transform == "edges":
                    # perform edge detection
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

                    # rebuild a VideoFrame, preserving timing information
                    new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    # return new_frame
                    self.result = new_frame
                elif self.transform == "long":
                    # wait a second.
                    asyncio.sleep(0.1)
                    
                    img = frame.to_ndarray(format="bgr24")
                    img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

                    # rebuild a VideoFrame, preserving timing information
                    new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    # return new_frame
                    self.result = new_frame
                elif self.transform == "rotate":
                    # rotate image
                    img = frame.to_ndarray(format="bgr24")
                    rows, cols, _ = img.shape
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                    img = cv2.warpAffine(img, M, (cols, rows))

                    # rebuild a VideoFrame, preserving timing information
                    new_frame = VideoFrame.from_ndarray(img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    # return new_frame
                    self.result = new_frame
                else:
                    # return frame
                    # self.result = new_frame
                    # print('no transform') # DEBUG
                    pass
            except Exception as e:
                print("EXCEPTION:", e)