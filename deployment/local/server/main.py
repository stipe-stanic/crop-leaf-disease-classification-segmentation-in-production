import io

import matplotlib.pyplot as plt
import torchvision
import uvicorn
import numpy as np
import torch
import joblib
import deployment.local.config as local_config

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from PIL import Image
from train.models import ResModel
from torch import Tensor
from img_aug_transform import CustomCLAHE


def custom_clahe_transform(img: Image) -> Image:
    """Apply custom Contrast Limited Adaptive Histogram Equalization (CLAHE) transformation to an image.

    :param img: The input image to be transformed.
    :returns: The transformed image.
    """

    transform = CustomCLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    return transform(img)


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
        x = x.to(device=package['device'])

        y_pred = model(x)

    y_pred = y_pred.cpu().numpy()

    return y_pred


class Model(str, Enum):
    ResModel = 'res-model'


app = FastAPI(
    title='ML Model',
    description='Model for classification of plant diseases',
    version='0.0.2',
)

app.add_middleware(CORSMiddleware, allow_origins=['*'])


@app.on_event('startup')
async def startup_event():
    """Initialize FastAPI and variables"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Pytorch using device: {device}')

    model = ResModel().to(device)
    model.load_state_dict(torch.load('../../../models_storage/export_models/ResModel.pth', map_location=device))
    model.eval()

    app.package = {
        'transform': joblib.load('../../../models_storage/export_models/transform.joblib'),
        'model': model,
        'device': device
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

    probabilities = np.exp(y) / np.sum(np.exp(y))
    predicted_class_index = np.argmax(probabilities)
    confidence_percentage = probabilities[predicted_class_index] * 100
    print(confidence_percentage)

    pred = local_config.class_names[predicted_class_index]

    return {
        'prediction': pred,
        'confidence': confidence_percentage
        }


if __name__ == '__main__':
    uvicorn.run(app, host=local_config.host, port=local_config.port)
