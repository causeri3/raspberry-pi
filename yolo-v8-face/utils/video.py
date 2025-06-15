import cv2
import logging
import numpy as np
import time

from utils.predict import Predict

class Stream:
    def __init__(self,
                 see_detection: bool = True,
                 available_devices: list | None = None):

        self.predict_class = Predict()

        self.frame = None
        self.available_devices = available_devices
        self.see_detection = see_detection
        self.face_size_threshold = 0.15
        self.rim_ratio = 0.2

    @staticmethod
    def return_camera_indexes():
        # checks the first 10 indexes.
        index = 0
        arr = []
        i = 10
        while i > 0:
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
                logging.debug("Found device under number {}.".format(index))
            index += 1
            i -= 1
        logging.info("Available devices found: {}".format(arr))
        return arr

    @staticmethod
    def choose_device(device_numbers: list):
        if len(device_numbers) < 2:
            return device_numbers[0]
        # default for my preferred set-up (no deeper meaning)
        elif len(device_numbers) == 3:
            return device_numbers[1]
        else:
            return device_numbers[-1]

    def predict_n_stream(self):


        combined_image_bytes, json_payload = self.predict_class.predict(
            self.frame,
            return_image=True)

        combined_image_array = np.frombuffer(combined_image_bytes, dtype=np.uint8)
        combined_img = cv2.imdecode(combined_image_array, flags=1)
        return combined_img, json_payload


    def draw_boxes(self):

        if not self.available_devices:
            self.available_devices = self.return_camera_indexes()

        device_numbers = self.choose_device(self.available_devices)
        # Initialize the webcam
        cap = cv2.VideoCapture(device_numbers)
        window_name = "Your camera, device no: {}".format(device_numbers)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        while cap.isOpened():
            start_time = time.time()

            # Read frame from the video
            ret, self.frame = cap.read()

            if not ret:
                break

            image, json_payload = self.predict_n_stream()
            if self.see_detection:
                cv2.imshow(window_name, image)
                # Press key esc to quit
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                logging.info(json_payload)

            logging.info("Processing one frame took {:.2f} sec".format(time.time() - start_time))

            best_crop = self.close_up_crop(json_payload)
            if best_crop is not None:
                break

        cap.release()
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
        return best_crop



    def close_up_crop(self, json_payload):
        if not json_payload:
            return None

        screen_h, screen_w, _ = self.frame.shape
        screen_area = screen_w * screen_h
        min_area = self.face_size_threshold * screen_area
        best_area = 0
        best_crop = None

        for f in json_payload['objects']:
            w = f['box']['width']
            h = f['box']['height']
            x = f['box']['x']
            y = f['box']['y']
            area = w * h
            if area >= min_area and area > best_area:
                if area >= min_area and area > best_area:
                    best_area = area
                    best_crop = self.crop_image((y,(y+h), x, (x+w)))

            return best_crop


    def crop_image(self,
                coordinates:list[float,float,float,float]):
        H, W, _ = self.frame.shape
        y1, y2, x1, x2 = map(int, map(round, coordinates))

        # give rim
        h = y2 - y1
        w = x2 - x1
        rim = int(self.rim_ratio * max(h, w))

        y1 = max(0, y1 - rim)
        x1 = max(0, x1 - rim)
        y2 = min(H, y2 + rim)
        x2 = min(W, x2 + rim)

        # make square
        h = y2 - y1
        w = x2 - x1
        if h != w:
            side = max(h, w)                 # target square side length

            # vertical padding needed?
            if h < side:
                pad = side - h
                top = max(0, y1 - pad // 2)
                bottom = min(H, top + side)
                y1, y2 = top, bottom

            # horizontal padding needed?
            if w < side:
                pad = side - w
                left = max(0, x1 - pad // 2)
                right = min(W, left + side)
                x1, x2 = left, right

            # If we got clipped on one edge, slide back so we really are square
            y2, x2 = min(y1 + side, H), min(x1 + side, W)
            y1, x1 = y2 - side, x2 - side

        return self.frame[y1:y2, x1:x2]





