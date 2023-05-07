import io
import os

import matplotlib.pyplot as plt
import torchvision
import uvicorn
import numpy as np
import torch
import joblib
import deployment.config as server_config

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from PIL import Image
from train.models import ResModel
from torch import Tensor


def preprocess(package: dict, input_image: Image) -> Tensor:
    transform = package['transform']

    img = transform(input_image)

    transform_back = torchvision.transforms.ToPILImage()
    image_processed = np.asarray(transform_back(img))
    plt.imshow(image_processed)
    plt.axis('off')
    plt.show()

    img_tensor = torch.unsqueeze(torch.FloatTensor(img), 0)

    return img_tensor


def predict(package: dict, input: Image) -> np.ndarray:
    x = preprocess(package, input)

    model = package['model']
    with torch.no_grad():
        x = x.to(device='cuda')

        y_pred = model(x)

    y_pred = y_pred.cpu().numpy()

    return y_pred


class Model(str, Enum):
    ResModel = 'res-model'


app = FastAPI(
    title='ML Model',
    description='Model for classification of plant diseases',
    version='0.0.1',
)

app.add_middleware(CORSMiddleware, allow_origins=['*'])


@app.on_event('startup')
async def startup_event():
    """Initialize FastAPI and variables"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Pytorch using device: {device}')

    model = ResModel().to(device)
    model.load_state_dict(torch.load('../model/ResModel.pth', map_location='cuda'))
    model.eval()

    app.package = {
        'transform': joblib.load('../model/transform.joblib'),
        'model': model
    }


@app.get("/")
def home():
    return "API is working as expected"


@app.post('/predict')
async def do_prediction(model: Model, file: UploadFile = File(...)) -> dict:
    filename = file.filename
    file_extension = filename.split('.')[-1] in ('jpg', 'png')
    if not file_extension:
        raise HTTPException(status_code=415, detail='Unsupported file provided.')

    content = await file.read()
    image = Image.open(io.BytesIO(content))

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    y = predict(app.package, image)[0]

    pred = ['apple_black_rot', 'apple_cedar_rust', 'apple_healthy', 'apple_scab'][y.argmax()]

    return {'prediction': pred}


if __name__ == '__main__':
    host = server_config.docker_host if os.getenv('docker-setup') else 'localhost'

    uvicorn.run(app, host=host, port=server_config.port)