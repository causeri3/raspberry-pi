import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
from PIL import Image

def transfer_style(content_image,
                   style_path=r"style-images/schalt.jpg",
                   model_path=r"model"):

    style_image = cv2.imread(style_path)
    size_threshold = 2000
    resizing_shape = (1000, 1000)
    content_shape = content_image.shape
    style_shape = style_image.shape

    if content_shape[0] > size_threshold or content_shape[1] > size_threshold:
        content_image = cv2.resize(content_image, resizing_shape)
    if style_shape[0] > size_threshold or style_shape[1] > size_threshold:
        style_image = cv2.resize(style_image, resizing_shape)

    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = tf.image.resize(style_image, (256, 256))

    hub_module = hub.load(model_path)
    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0].numpy()
    stylized_image = stylized_image[0] * 255.0
    stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
    return stylized_image


def generate_image_list(content_image):
    frames = []

    for i in range(10):
        print(f"Iteration {i + 1}")
        content_image = transfer_style(content_image)

        img_rgb = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(img_rgb))
