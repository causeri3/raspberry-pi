# Selfusion
### Repository for an art installation  
*(made for Fusion 2025)*

---

Takes automated selfies, generates more pictures from that, and plays a bouncing gif.

---

Using:
- **Automated Selfies**: Yolov8 trained by [arnabdhar](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection)
- **Offline image generation**: Neural Style Transfer [deepeshdm](https://github.com/deepeshdm/PixelMix/tree/main)
- **Online image generation**: [Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo) in repo [sdxlturbo-api](https://github.com/causeri3/sdxlturbo-api)

---
## Settings / Args
See all arguments:
* with uv `uv run main.py --help`
* standard `python main.py --help`

Most interesting settings you will find in `selfusion_utils.args`, such as:  
- gif delay  
- waiting time  
- loading bar time  
- come closer screen or not
- face size threshold
- yolo settings (confidence, IoU)
- diffuser settings (strength, inference steps, prompt ...)


---
Another bit that's might be fun: Adding more style pictures — they are picked randomly.  
You can find them in `neural_style_transfer.style_images`.

---
## Hardware
- Raspberry Pi 5 (16GB)

---
## Software
### Dependencies

#### SDXLTurbo API
It's meant to work with [sdxlturbo-api](https://github.com/causeri3/sdxlturbo-api) hosted on some machine (runs well with mps on my Mac or cuda on a cloud VM with GPU - if you want you can test just the sdxlturbo code straight on Colab with T4, was super easy).

Anyway, it still has offline functionality. If it cannot get results from the Stable diffusion API, it generates pictures with neural style transfer on the local machine / the pi.


#### Linux

```bash
sudo apt-get install libgtk2.0-dev pkg-config
```

#### Python
The raspberry had `python 3.11` preinstalled

```bash
pip install -r requirements.txt
```
Gets it running.

Or even better if you use [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1), run in the directory of this repo:
```sh
uv venv --python 3.11
uv pip install -r requirements.txt
```
**Note**:
Some of the code in this repo was used for experimentation (such as the `stable-diffusion` folder or `yolo_onnx_openvino.py`, they need more dependencies, so most directories have their own requirements files)

---
## Run
* with uv
 `uv run selfusion.py`

* standard
 `python selfusion.py`

### Systemd

I run it as systemd service:
```
[Unit]
Description=Fusion Raspberry Process
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=/home/pi/src/raspberry-pi
ExecStart=/home/pi/src/selfusion_venv/bin/python /home/pi/src/selfusion/selfusion.py
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/pi/.Xauthority

[Install]
WantedBy=default.target
```
that file goes under:
`~/.config/systemd/user/fusion.service`

**run service**
```bash
systemctl --user daemon-reload
systemctl --user enable fusion.service
systemctl --user start fusion.service
```



---
This project is licensed under AGPL v3+