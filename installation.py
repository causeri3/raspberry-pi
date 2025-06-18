import logging
from yolo_v8_face.utils.video import Stream
import cv2
import threading
import numpy as np
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

def processing_worker(selfie):
    sdxl_result = {}
    sdxl_thread = threading.Thread(target=request_sdxlturbo, args=(selfie, sdxl_result))
    sdxl_thread.start()

    logging.info("Generating NST images...")
    nst_images = generate_image_list(selfie)

    sdxl_thread.join(timeout=600)

    if sdxl_result.get('success', False):
        gif_frames = [img for img in sdxl_result['images']]
    else:
        gif_frames = [np.array(img) for img in nst_images]

    with gif_lock:
        global frames
        frames = gif_frames
        new_frames_event.set()

def selfie_capture_worker():
    global selfie_image
    stream = Stream(see_detection=False, available_devices=[0])

    selfie = stream.draw_boxes()  # This is blocking, but now it runs in a thread
    logging.info("Selfie captured.")

    with selfie_lock:
        selfie_image = selfie
        selfie_ready_event.set()

def main_loop():
    frame_index = 0
    is_generating = False

    # Start first selfie capture thread
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
                frame = frames[frame_index % len(frames)]
                frame_index += 1
            else:
                frame = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

        cv2.imshow("TRANSFORMATION", frame)
        key = cv2.waitKey(100)
        if key == 27:  # ESC to quit
            break

        # If new gif frames ready → start new selfie capture
        if new_frames_event.is_set():
            is_generating = False
            new_frames_event.clear()
            threading.Thread(target=selfie_capture_worker).start()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
