import logging
from yolo_v8_face.utils.video import Stream
import cv2
import threading
import numpy as np
import time

from neural_style_transfer.nst import generate_image_list
from installation_utils.utils import (request_sdxlturbo
                                      )
# from utils.args import get_args
#
# args, unknown = get_args()


WIDTH, HEIGHT   = (512, 512)

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

def request_sdxlturbo_with_flag(selfie, result_container, stop_flag):
    request_sdxlturbo(selfie, result_container)
    if result_container.get('success', False):
        stop_flag.set()

def processing_worker(selfie):
    try:
        sdxl_result = {}
        stop_flag = threading.Event()

        sdxl_thread = threading.Thread(
            target=request_sdxlturbo_with_flag,
            args=(selfie, sdxl_result, stop_flag))
        sdxl_thread.start()

        logging.info("Generating NST images...")
        nst_images = generate_image_list(selfie, should_stop=stop_flag.is_set)
        sdxl_thread.join(timeout=180)

        if sdxl_result.get('success', False):
            gif_frames = [img for img in sdxl_result['images']]
        else:
            gif_frames = [np.array(img) for img in nst_images]

        with gif_lock:
            global frames
            frames = gif_frames
            new_frames_event.set()
    except Exception as e:
        logging.exception("processing_worker crashed! Exiting whole process...")
        import os
        os._exit(1)

def selfie_capture_worker():
    global selfie_image
    stream = Stream(see_detection=False, available_devices=[0])

    selfie = stream.draw_boxes()
    logging.info("Selfie captured.")
    logging.info(f"Selfie capture result: {type(selfie)}, None? {selfie is None}")

    with selfie_lock:
        selfie_image = selfie
        selfie_ready_event.set()

def main_loop():
    is_generating = False
    fps = 1
    pause = 1
    next_selfie_time = 0
    longer_delay_ms = max(int(pause * 1000), 1)

    threading.Thread(target=selfie_capture_worker).start()

    while True:
        # If selfie ready and no processing running → start processing worker
        if selfie_ready_event.is_set() and not is_generating:
            with selfie_lock:
                selfie = selfie_image
            threading.Thread(target=processing_worker, args=(selfie,)).start()
            is_generating = True
            selfie_ready_event.clear()

        # Draw gif frame or noise
        with gif_lock:
            if frames:
                frames_bgr = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in frames]
                bounce_frames = frames_bgr + frames_bgr[-2::-1]
                delay_ms = max(int(1000 / fps), 1)
            else:
                bounce_frames = [np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)]
                delay_ms = 0

        for i, frame in enumerate(bounce_frames):
            cv2.imshow("TRANSFORMATION", frame)

            is_turning_point = (i == 0 or i == len(bounce_frames) // 2 or i == len(bounce_frames) - 1)
            wait = longer_delay_ms if is_turning_point else delay_ms

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
                logging.info(f"Waiting")

                if time.time() >= next_selfie_time:
                    threading.Thread(target=selfie_capture_worker).start()
                    next_selfie_time = 0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
