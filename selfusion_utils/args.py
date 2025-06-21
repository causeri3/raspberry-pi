from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-cam',
                        '--cam-device-number',
                        required=False,
                        default=None,
                        type=int,
                        help='Overwrite camera device number (integer)')

    parser.add_argument('-sd',
                        '--see-detection',
                        required=False,
                        action='store_true',
                        help="""See object detection in streamed video output.
                        If argument in command, you will see the video stream with bounding boxes""")

    parser.add_argument('-conf',
                        '--confidence-threshold',
                        required=False,
                        default=0.8,
                        type=float,
                        help="""Confidence threshold for detected object (float between 0 and 1)""")

    parser.add_argument('-iou',
                        '--iou-threshold',
                        required=False,
                        default=0.05,
                        type=float,
                        help="""Threshold for IoU - Intersection over Union (float between 0 and 1)""")

    parser.add_argument('-close',
                        '--face-size-threshold',
                        required=False,
                        default=.15,
                        type=float,
                        help="""how many % of the screen does the face need to fill for a selfie to be taken - float between 0 and 1""")

    parser.add_argument('-fps',
                        '--fps',
                        required=False,
                        default=4,
                        type=int,
                        help='Frames per second for gif display')

    parser.add_argument('-gp',
                        '--gif-pause-sec',
                        required=False,
                        default=1,
                        type=int,
                        help='Seconds to pause with last pic')

    parser.add_argument('-w',
                        '--wait-sec',
                        required=False,
                        default=45,
                        type=int,
                        help='Waiting time before yolo approaches to take another selfie')

    parser.add_argument('-ld',
                        '--loading-duration-sec',
                        required=False,
                        default=130,
                        type=int,
                        help='How long the loading bar takes')

    parser.add_argument('-p',
                        '--prompt',
                        required=False,
                        default="dmt",
                        type=str,
                        help='Prompt')

    parser.add_argument('-cc',
                        '--come-closer-screen',
                        required=False,
                        action='store_false',
                        help="""
                        If argument in command, 
                        you will see a come closer screen between picture generation, 
                        otherwise the old gif will keep playing""")

    parser.add_argument('-n',
                        '--amount-pics',
                        required=False,
                        default=16,
                        type=int,
                        help='Number of pictures to generate')

    parser.add_argument('-steps',
                        '--num-inference-steps',
                        required=False,
                        default=50,
                        type=int,
                        help='Number of inference steps per image')

    parser.add_argument('-smin',
                        '--strength-min',
                        required=False,
                        default=0.05,
                        type=float,
                        help='Minimum strength for image transformation (0-1)')

    parser.add_argument('-smax',
                        '--strength-max',
                        required=False,
                        default=0.35,
                        type=float,
                        help='Maximum strength for image transformation (0-1)')

    parser.add_argument('-gs',
                        '--guidance-scale',
                        required=False,
                        default=8,
                        type=int,
                        help='Guidance scale for diffusion model')

    # to not get into trouble with uvicorn args
    args = parser.parse_known_args()
    return args