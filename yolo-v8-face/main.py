import logging
from utils.video import Stream
import cv2
from PIL import Image
import datetime
from utils.transformation_screen import show_loader, play_gif_frames
# from utils.args import get_args
#
# args, unknown = get_args()

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

# if args.cam_device_number:
#     logging.info("""You chose camera device no: {}""".format(args.cam_device_number))
#     device = [args.cam_device_number]
# else:
#     device = None
# window_name = 'SELFIE'
# PROMPT = "dmt"

stream = Stream(
    #see_detection=args.see_detection,
    #available_devices = [0],
    see_detection=True,
    available_devices=None
    )
selfie = stream.draw_boxes()
#cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
#cv2.imshow(window_name, selfie)
# cv2.waitKey(1)
# cv2.destroyWindow(window_name)
# cv2.waitKey(1)

stamp = datetime.now().strftime("%-d_%B_%H_%M").lower()
cv2.imwrite("selfie" + '_' + stamp + ".png", selfie)