from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections
from PIL import Image

repo_name="arnabdhar/YOLOv8-Face-Detection"
model_file_name="model.pt"
model_path = hf_hub_download(repo_id=repo_name, filename=model_file_name)
model = YOLO(model_path)
model.export(format="openvino")


model = YOLO(".cache/huggingface/hub/models--arnabdhar--YOLOv8-Face-Detection/snapshots/52fa54977207fa4f021de949b515fb19dcab4488/model_openvino_model")
model = YOLO(".cache/huggingface/hub/models--arnabdhar--YOLOv8-Face-Detection/snapshots/52fa54977207fa4f021de949b515fb19dcab4488/model.onnx")

image_path = "Downloads/img1.JPG"
pil_image = Image.open(image_path)
output = model(pil_image)
results = Detections.from_ultralytics(output[0])
x1, y1, x2, y2 = results.xyxy[0]