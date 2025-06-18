import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import logging
import time
import random
import os

REZ = (1000, 1000)
AMOUNT_PICS = 5

def get_random_style_image(style_folder=r"neural_style_transfer/style_images"):
    all_files = os.listdir(style_folder)
    jpg_files = [f for f in all_files if f.lower().endswith('.jpg')]
    random_file = random.choice(jpg_files)
    style_path = os.path.join(style_folder, random_file)
    return style_path


def transfer_style(content_image,
                   style_path,
                   model_path=r"neural_style_transfer/model"):

    style_image = cv2.imread(style_path)
    content_image = cv2.resize(content_image, REZ)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.

    hub_module = hub.load(model_path)
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()
    stylized_image = stylized_image[0] * 255.0
    stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
    return stylized_image


def generate_image_list(content_image):
    style_path = get_random_style_image()
    frames = []
    start_time = time.time()
    for i in range(AMOUNT_PICS):
        print(f"Iteration {i + 1}")
        content_image = transfer_style(content_image, style_path)

        img_rgb = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(img_rgb))
    logging.info("Stylising {} pics took {:.2f} sec".format(AMOUNT_PICS, time.time() - start_time))

    return frames

