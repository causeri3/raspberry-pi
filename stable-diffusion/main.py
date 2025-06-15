from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch
from PIL import ImageOps
from time import time
import logging
from pathlib import Path
from datetime import datetime
import math

image_path = "path"
# ___________ LOAD MODEL _________________ #
pipe = AutoPipelineForImage2Image.from_pretrained("sd-legacy/stable-diffusion-v1-5",
                                                  #torch_dtype=torch.float32,
                                                  low_cpu_mem_usage=True,
                                                  #variant="fp16",
                                                  #cache_dir="/models-cache"
                                                  )
pipe.to("cpu")


# ___________ PARAMS _________________ #
# When using SDXL-Turbo for image-to-image generation, make sure that num_inference_steps * strength is larger or equal to 1.
# The image-to-image pipeline will run for int(num_inference_steps * strength) steps, e.g. 0.5 * 2.0 = 1 step in our example below.
NUM_INFERENCE_STEPS=15
STRENGTH = 0.5
GUIDANCE_SCALE=8
PROMPT = "dmt"

# mdma
# amphetamin
# methamphetamin
# heroin
# cocaine
# ketamine
# cannabis
# dmt
# lsd
# alcohol

# double exposure
# double exposure movement
# detailed scene
# 8k
# glitch art
# cinematic
# DMT
# superrealistic

REZ = (512, 512)
# ___________ YALLA _________________ #
images_list = []
init_image = load_image(image_path).resize(REZ)

images_list.append(init_image)

start_time = time()
image = pipe(PROMPT,
             image=init_image,
             # num_inference_steps=math.ceil(1/STRENGTH_MIN),
             num_inference_steps=NUM_INFERENCE_STEPS,
             strength=STRENGTH,
             guidance_scale=GUIDANCE_SCALE).images[0]
print("It took {:.2f} Sec".format((time() - start_time)))
images_list.append(image)

image.show()

def save_images_gif(images_list, prompt=PROMPT):
    stamp = datetime.now().strftime("%-d_%B_%H_%M").lower()
    folder_name = (
            f"steps{NUM_INFERENCE_STEPS}"
            + f"_strength{STRENGTH}"
            + f"_scale{GUIDANCE_SCALE}"
            + f"_prompt_{prompt}"
            + '_' + stamp
    )

    out_dir = Path("images") / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    first, *rest = images_list

    first.save(
        out_dir / "animation.gif",
        save_all=True,  # multi-frame file
        append_images=rest,
        duration=500,  # ms per frame  ──► 0.5 s
        loop=0,  # 0 infinite
        optimize=True,  # lossless LZW + palette squeeze
        disposal=2  # clear before drawing next frame (prevents trails)
    )