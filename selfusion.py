import logging

from selfusion_utils.transformation import Transformation
from selfusion_utils.args import get_args

args, unknown = get_args()

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

if __name__ == "__main__":
    Transformation().run()