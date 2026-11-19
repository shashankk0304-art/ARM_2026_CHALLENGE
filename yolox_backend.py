"""OpenCV Zoo YOLOX COCO detector wrapper for Arm CPU inference.

Model and post-processing follow OpenCV Zoo's object_detection_yolox example.
The model detects COCO objects; it is not a waste-trained detector.
"""
import cv2
import numpy as np

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


class YoloX:
    def __init__(self, model_path, conf_threshold=0.35, nms_threshold=0.5):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = 640
        self.strides = (8, 16, 32)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        grids, strides = [], []
        for stride in self.strides:
            h = w = self.input_size // stride
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grids.append(np.stack((xv, yv), axis=2).reshape(1, -1, 2))
            strides.append(np.full((1, h * w, 1), stride))
        self.grids = np.concatenate(grids, axis=1).astype(np.float32)
        self.expanded_strides = np.concatenate(strides, axis=1).astype(np.float32)

    def infer(self, letterboxed_rgb):
        blob = np.ascontiguousarray(letterboxed_rgb.transpose(2, 0, 1)[None], dtype=np.float32)
        self.net.setInput(blob)
        raw = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        dets = raw.copy()
        dets[:, :2] = (dets[:, :2] + self.grids[0]) * self.expanded_strides[0]
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self.expanded_strides[0]
        boxes = dets[:, :4]
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        scores_by_class = dets[:, 4:5] * dets[:, 5:]
        scores = scores_by_class.max(axis=1)
        class_ids = scores_by_class.argmax(axis=1)
        keep = cv2.dnn.NMSBoxesBatched(xyxy.tolist(), scores.tolist(), class_ids.tolist(),
                                       self.conf_threshold, self.nms_threshold)
        if len(keep) == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack((xyxy, scores, class_ids))[np.asarray(keep).reshape(-1)]
