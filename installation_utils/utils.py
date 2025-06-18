from PIL import Image
import requests
import io
import numpy as np
import logging
import cv2

def request_sdxlturbo(selfie, result_container):
    try:
        files = selfie_to_file_data(selfie)
        data = {'prompt': 'dmt'}

        response = requests.post("http://localhost:8000/sdxlturbo",
                                  files=files,
                                  data=data,
                                  timeout=600)
        buffer = io.BytesIO(response.content)
        result_container['images'] = np.load(buffer)
        result_container['success'] = True
    except Exception as e:
        logging.warning(f"SDXL Turbo failed: {e}")
        result_container['success'] = False


def selfie_to_file_data(selfie):
    selfie_rgb = Image.fromarray(selfie[..., ::-1])
    buffer = io.BytesIO()
    selfie_rgb.save(buffer, format="PNG")
    buffer.seek(0)
    return {'file': ('selfie.png', buffer, 'image/png')}


def play_gif_frames(
        image_list:list[np.ndarray],
        window_name='TRANSFORMATION',
        fps=6,                 # frames per second
        pause: float = 2,
        loop=True):             # replay forever until Esc, or just once

    delay_ms = max(int(1000 / fps), 1)
    longer_delay_ms = max(int(pause * 1000), 1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frames = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL → RGB ndarray → BGR
              for img in image_list]

    bounce_frames = frames + frames[-2::-1]

    while True:
        for i, frame in enumerate(bounce_frames):
            cv2.imshow(window_name, frame)
            is_turning_point = (i == 0 or i == len(frames) - 1)
            wait = longer_delay_ms if is_turning_point else delay_ms
            if cv2.waitKey(wait) & 0xFF == 27:   # Esc
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
                break
        if not loop:
            break

