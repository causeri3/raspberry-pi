from PIL import Image
import requests
import io
import numpy as np
import logging

def request_sdxlturbo(selfie,
                      result_container,
                      prompt='dmt',
                      # 10 min, but thread kills it anyway earlier with LOADING_DURATION_SEC
                      timeout=600):
    try:
        files = _selfie_to_file_data(selfie)
        data = {'prompt': prompt}

        response = requests.post("http://localhost:8000/sdxlturbo",
                                  files=files,
                                  data=data,
                                  timeout=timeout)
        buffer = io.BytesIO(response.content)
        result_container['images'] = np.load(buffer)
        result_container['success'] = True
    except Exception as e:
        logging.warning(f"SDXL Turbo failed: {e}")
        result_container['success'] = False


def _selfie_to_file_data(selfie):
    selfie_rgb = Image.fromarray(selfie[..., ::-1])
    buffer = io.BytesIO()
    selfie_rgb.save(buffer, format="PNG")
    buffer.seek(0)
    return {'file': ('selfie.png', buffer, 'image/png')}
