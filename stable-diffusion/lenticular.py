import cv2
import numpy as np
import time
import math

img1 = cv2.imread("/Downloads/img1.jpg").astype(np.float32) / 255.0
img2 = cv2.imread("/Downloads/img2.jpg").astype(np.float32) / 255.0

height, width, channels = img1.shape

blend_height = 200  # pixels
speed = 0.7

while True:
    t = time.time()
    # compute vertical center of blending belt
    center = int((height - blend_height) * (0.5 + 0.5 * math.sin(t * speed)))

    # initialize empty frame
    blended = np.zeros_like(img1)

    # fully img1 above blending belt
    if center > 0:
        blended[:center] = img1[:center]

    # fully img2 below blending belt
    if center + blend_height < height:
        blended[center+blend_height:] = img2[center+blend_height:]

    # blend inside belt
    for i in range(blend_height):
        y = center + i
        alpha = i / blend_height  # linear blending from img1 (top) to img2 (bottom)
        blended[y] = (1 - alpha) * img1[y] + alpha * img2[y]

    cv2.imshow("Lenticular", (blended * 255).astype(np.uint8))
    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()
