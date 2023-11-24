
import numpy as np
import argparse
import torch
import cv2
from ultralytics import YOLO

class BoundingBoxDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_bounding_boxes(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)
        boxes_data = results[0].boxes.data.cpu().numpy()
        boxes = boxes_data[:, :4]
        confidences = boxes_data[:, 4]
        # Additional processing if necessary
        return boxes, confidences  # Modify this if you need additional info like confidences


def load_trained_model(model_path):
    # Load the trained YOLO model
    model = YOLO(model_path)  # Assuming your model file is named 'yolov8m_80ep.pt'
    return model

def apply_nms(boxes, confidences, score_threshold=0.4, nms_threshold=0.4):
    # Convert boxes to a format that cv2.dnn.NMSBoxes can work with
    boxes_array = np.array([[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes])

    # Get the indices of the boxes to keep
    indices = cv2.dnn.NMSBoxes(boxes_array.tolist(), confidences, score_threshold, nms_threshold)
    # Flatten the list of lists
    indices = indices.tolist()

    return indices