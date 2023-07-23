# NOTE not the best codebase. could use some refactoring, when technical debt has accumulated too much

# basics
import numpy as np
from matplotlib import pyplot as plt
# computer vision
import cv2
from PIL import Image
import onnxruntime
# local modules

# NOTE absolute import from server-level scripts (for prod)
from functions.object.classes import CLASSES 
from functions.object.utils import nms, compute_iou, xywh2xyxy

# # NOTE absolute import from package (for dev)
# from server.functions.object.classes import CLASSES 
# from server.functions.object.utils import nms, compute_iou, xywh2xyxy

# NOTE utils func ideas: letterbox, scale_coords, plot_one_box


opt_options = onnxruntime.SessionOptions()
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# nano
# model_path = './models/yolov8n_640.onnx' # when used as a module
# model_path2 = './server/functions/object/models/yolov8n_640.onnx' # when used as a script (dev)
# medium
model_path = './models/yolov8m_640.onnx' # when used as a module
model_path2 = './functions/object/models/yolov8m_640.onnx' # when used as a script (dev)

try: 
    session = onnxruntime.InferenceSession(model_path, opt_options, providers=EP_list)
    # print(1)
except Exception as e: 
    session = onnxruntime.InferenceSession(model_path2, opt_options, providers=EP_list)
    # print(2)
    


class ObjectDetector:

    def __init__(self):
        self.session = session
        # model I/O
        self.input_names = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None
        self.initialize_yolo()


    def initialize_yolo(self):
        # inputs
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        # outputs
        model_output = self.session.get_outputs()
        self.output_names = [model_output[i].name for i in range(len(model_output))]
        self.output_shape = model_output[0].shape


    def info(self):
        print('input_names', self.input_names)
        print('input_shape', self.input_shape)
        print('output_names', self.output_names)
        print('output_shape', self.output_shape)


    def detect(self, image):
        image_height, image_width = image.shape[:2]
        input_height, input_width = self.input_shape[2:]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (input_width, input_height))

        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        # print(input_tensor.shape) # DEBUG

        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        
        predictions = np.squeeze(outputs).T

        conf_thresold = 0.8
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_thresold, :]
        scores = scores[scores > conf_thresold]
        # print(scores) # DEBUG

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        # print(class_ids) # DEBUG

        # Get bounding boxes for each object
        boxes = predictions[:, :4]

        #rescale box
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)
        # print(boxes) # DEBUG

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, 0.3)
        # print(indices) # DEBUG

        # print(boxes[indices], scores[indices], class_ids[indices]) # DEBUG
        # result = (boxes, scores, class_ids, indices)
        result = {'boxes': boxes, 'scores': scores, 'class_ids': class_ids, 'indices': indices}
        return result



def draw(image, boxes, scores, class_ids, indices):
    image = image.copy()
    for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
        bbox = bbox.round().astype(np.int32).tolist()
        cls_id = int(label)
        cls = CLASSES[cls_id]
        color = (0,255,0)
        cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv2.putText(image,
                    f'{cls}:{int(score*100)}', (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, [225, 255, 255],
                    thickness=1)
    return image


def main():
    object = ObjectDetector()
    image = cv2.imread('./tests/test_images/test_guy.jpg')
    result = object.detect(image)
    drawn = draw(image, *result)
    plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
    plt.show()
    print()

if __name__ == "__main__": main()