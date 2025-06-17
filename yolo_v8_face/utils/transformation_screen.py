import cv2
import numpy as np
# from utils.sdxlturbo import REZ

REZ = (512, 512)

WIDTH, HEIGHT   = REZ

DURATION_SEC    = 210
FPS             = 30
BAR_HEIGHT_PX   = 60
BORDER_THICK    = 1

TOTAL_FRAMES    = int(DURATION_SEC * FPS)
BAR_TOTAL_WIDTH = WIDTH // 3
LEFT_EDGE_X     = (WIDTH - BAR_TOTAL_WIDTH) // 2
TOP_EDGE_Y      = (HEIGHT - BAR_HEIGHT_PX) // 2
INNER_WIDTH     = BAR_TOTAL_WIDTH - 2 * BORDER_THICK
PX_PER_FRAME    = INNER_WIDTH / TOTAL_FRAMES
FRAME_DELAY_MS  = int(1000 / FPS)

outer_tl = (LEFT_EDGE_X, TOP_EDGE_Y)
outer_br = (LEFT_EDGE_X + BAR_TOTAL_WIDTH,
            TOP_EDGE_Y + BAR_HEIGHT_PX)
inner_tl = (outer_tl[0] + BORDER_THICK,
            outer_tl[1] + BORDER_THICK)
inner_max_br_y = outer_br[1] - BORDER_THICK

def show_loader(stop_event,
                window_name='LOADING'
                ):

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH, HEIGHT)

    f = 0
    while not stop_event.is_set():
        # ---------------------------------------------------
        frame = np.random.randint(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)

        # static outline
        cv2.rectangle(frame, outer_tl, outer_br, (255, 255, 255), BORDER_THICK)

        # growing fill
        fill_progress = min(f, TOTAL_FRAMES)
        fill_w        = int(PX_PER_FRAME * fill_progress)
        fill_br = (inner_tl[0] + fill_w, inner_max_br_y)
        cv2.rectangle(frame, inner_tl, fill_br, (255, 255, 255), -1)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(FRAME_DELAY_MS) & 0xFF == 27:   # Esc pressed?
            stop_event.set()
            break
        f += 1
        # ---------------------------------------------------

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)



def play_gif_frames(
        image_list:list[np.ndarray],
        window_name='TRANSFORMATION',
        fps=6,                 # frames per second
        last_pause: float = 1.5,
        loop=True):             # replay forever until Esc, or just once

    delay_ms = max(int(1000 / fps), 1)
    last_delay_ms = max(int(last_pause * 1000), 1)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frames = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL → RGB ndarray → BGR
              for img in image_list]

    while True:
        for i, frame in enumerate(frames):
            cv2.imshow(window_name, frame)
            wait = last_delay_ms if i == len(frames) - 1 else delay_ms
            if cv2.waitKey(wait) & 0xFF == 27:   # Esc
                cv2.destroyWindow(window_name)
                cv2.waitKey(1)
                break
        if not loop:
            break
