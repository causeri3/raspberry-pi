import numpy as np
import logging
import time
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from PIL import Image

from yolo_v8_face.utils.payloads import json_payload, image_payload
from selfusion_utils.args import get_args


args, unknown = get_args()

class Predict:
    def __init__(self,
                 repo_name: str ="arnabdhar/YOLOv8-Face-Detection",
                 model_file_name: str = "model.pt",
                 confidence_threshold: float = args.confidence_threshold,
                 iou_threshold:float = args.iou_threshold):

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.repo_name= repo_name
        self.model_file_name = model_file_name
        self.model = self.load_model()

    @staticmethod
    def convert_to_model_format(frame: np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def load_model(self):
        model_path = hf_hub_download(repo_id=self.repo_name, filename= self.model_file_name)
        model = YOLO(model_path)
        return model

    def predict(self,
                image: np.ndarray,
                return_image: bool = False):

        """Gives object detection for one image from yolo v7 via onnx runtime session
        :param image: Input image in bytes or as np array
        :param return_image: Returning image with labels, confidences and boxes as bytes
        :returns If none of the above return options is chosen the image will open locally"""

        start_time = time.time()
        pil_image = self.convert_to_model_format(image)
        output = self.model(pil_image,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold)
        detected_objects = Detections.from_ultralytics(output[0])

        logging.debug("Detected {} objects".format(len(detected_objects)))

        json_output = json_payload(detected_objects)
        end_time = time.time()
        logging.debug("One detection event took {:.2f} seconds".format(end_time - start_time))
        if return_image:
            bytes_output = image_payload(detected_objects, image)
            return bytes_output, json_output

        return json_output



