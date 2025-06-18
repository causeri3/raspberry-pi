import logging
from yolo_v8_face.utils.video import Stream
import cv2
import threading
import numpy as np
import time
import os

from neural_style_transfer.nst import generate_image_list
from installation_utils.utils import (request_sdxlturbo,
                                      draw_loading_bar,
                                      draw_come_closer)
# from utils.args import get_args
#
# args, unknown = get_args()

WIDTH, HEIGHT  = (1280, 1024)

FPS = 1
PAUSE = 1
LONGER_DELAY_MS = max(int(PAUSE * 1000), 1)
LOADING_DURATION_SEC = 130

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

# if args.cam_device_number:
#     logging.info("""You chose camera device no: {}""".format(args.cam_device_number))
#     device = [args.cam_device_number]
# else:
#     device = None
# window_name = 'SELFIE'
# PROMPT = "dmt"


frames = []
gif_lock = threading.Lock()
new_frames_event = threading.Event()
selfie_lock = threading.Lock()
selfie_ready_event = threading.Event()
selfie_image = None
wait_screen = True

def request_sdxlturbo_with_flag(selfie, result_container, stop_flag):
    request_sdxlturbo(selfie, result_container)
    if result_container.get('success', False):
        stop_flag.set()

def processing_worker(selfie):
    try:
        sdxl_result = {}
        stop_flag = threading.Event()

        sdxl_start = time.time()
        sdxl_thread = threading.Thread(
            target=request_sdxlturbo_with_flag,
            args=(selfie, sdxl_result, stop_flag))
        sdxl_thread.start()

        logging.info("Generating NST images...")
        nst_images = generate_image_list(selfie, should_stop=stop_flag.is_set)
        sdxl_thread.join(timeout=LOADING_DURATION_SEC)

        if sdxl_result.get('success', False):
            logging.info(f"SDXLTurbo took {time.time() - sdxl_start} sec")
            gif_frames = [img for img in sdxl_result['images']]
        else:
            gif_frames = [np.array(img) for img in nst_images]

        with gif_lock:
            global frames
            global wait_screen
            frames = gif_frames
            wait_screen = False
            new_frames_event.set()
    except Exception as e:
        logging.exception(f"processing_worker crashed with Exception {e}")
        os._exit(1)

def selfie_capture_worker():
    global selfie_image
    stream = Stream(see_detection=False, available_devices=[0])

    selfie = stream.draw_boxes()
    logging.info("Selfie captured.")
    # logging.info(f"Selfie capture result: {type(selfie)}, None? {selfie is None}")

    with selfie_lock:
        selfie_image = selfie
        selfie_ready_event.set()

def main_loop(come_closer_screen=False):
    is_generating = False
    global wait_screen
    next_selfie_time = 0
    loading_start_time = 0

    threading.Thread(target=selfie_capture_worker).start()

    while True:
        # If selfie ready and no processing running → start processing worker
        if selfie_ready_event.is_set() and not is_generating:
            with selfie_lock:
                selfie = selfie_image
            threading.Thread(target=processing_worker, args=(selfie,)).start()
            is_generating = True
            loading_start_time = time.time()
            selfie_ready_event.clear()

        # Draw gif frame or noise
        with gif_lock:
            if frames and not wait_screen:
                frames_bgr = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in frames]
                bounce_frames = frames_bgr + frames_bgr[-2::-1]
                delay_ms = max(int(1000 / FPS), 1)
            else:
                bounce_frames = [np.random.randint(0, 256, (HEIGHT, HEIGHT, 3), dtype=np.uint8)]
                delay_ms = 0

        for i, frame in enumerate(bounce_frames):
            frame_copy = frame.copy()

            # Draw loading bar if generating
            if is_generating:
                progress = min(1.0, (time.time() - loading_start_time) / LOADING_DURATION_SEC)
                draw_loading_bar(frame_copy, progress)

            elif wait_screen:
                draw_come_closer(frame_copy)

            frame_resized = cv2.resize(frame_copy, (HEIGHT, HEIGHT), interpolation=cv2.INTER_LINEAR)

            # black background
            canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            # centering the square frame
            x_offset = (WIDTH - HEIGHT) // 2
            y_offset = (HEIGHT - HEIGHT) // 2

            # square frame onto the background
            canvas[y_offset:y_offset + HEIGHT, x_offset:x_offset + HEIGHT] = frame_resized

            cv2.imshow("TRANSFORMATION", canvas)
            # cv2.imshow("TRANSFORMATION", frame_resized)
            # cv2.imshow("TRANSFORMATION", frame_copy)

            is_turning_point = (i == 0 or i == len(bounce_frames) // 2 or i == len(bounce_frames) - 1)
            wait = LONGER_DELAY_MS if is_turning_point else delay_ms

            key = cv2.waitKey(wait)
            if key == 27:  # ESC to quit
                cv2.destroyAllWindows()
                return

            if new_frames_event.is_set():
                is_generating = False
                new_frames_event.clear()
                # wait for 30 sec
                next_selfie_time = time.time() + 30

            if not selfie_ready_event.is_set() and not is_generating and next_selfie_time != 0:
                wait_screen = False
                is_generating = False
                logging.info(f"Waiting")

                if time.time() >= next_selfie_time:
                    if come_closer_screen:
                        wait_screen = True
                    threading.Thread(target=selfie_capture_worker).start()
                    next_selfie_time = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        logging.exception(f"Min loop crashed with Exception {e}")
        os._exit(1)