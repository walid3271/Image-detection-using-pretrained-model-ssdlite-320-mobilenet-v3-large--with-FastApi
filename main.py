from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

app = FastAPI()

model = ssdlite320_mobilenet_v3_large(pretrained=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

CLASSES = [
    'background', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
    'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
    'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@app.post("/detect")
async def detect(file: UploadFile = File(...), threshold: float = 0.80):
    image = Image.open(io.BytesIO(await file.read()))
    image_tensor = ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)
        print(predictions[0])

    detection_results = {}
    for idx in range(len(predictions[0]['labels'])):
        if predictions[0]['scores'][idx] >= threshold:
            label = CLASSES[int(predictions[0]['labels'][idx].cpu().numpy())]
            score = predictions[0]['scores'][idx].cpu().numpy().item()
            box = predictions[0]['boxes'][idx].cpu().numpy().tolist()
            res = {label:{'score': score, 'box': box}}
            detection_results.update(res)
            
    return JSONResponse(detection_results)

# python -m uvicorn main:app --reload