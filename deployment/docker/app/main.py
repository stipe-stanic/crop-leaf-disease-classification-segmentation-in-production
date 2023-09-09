import io
import sys
import numpy as np
import torch

from app.config import class_names
from app.res_model import ResModel
from app.exception_handler import python_exception_handler
from app.segmentation.segment import segment_object
from app.util.preprocessing import preprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from PIL import Image


def predict(package: dict, input: Image) -> np.ndarray:
    """Run prediction using a provided model on a preprocessed input image.

    :param package: A dictionary containing the 'model', 'device', and 'transform' objects.
    :param input: The input image to be classified.
    :returns: The prediction values for each class as a numpy array.
    """

    x = segment_object(input, package['segment_model_path'])
    x = preprocess(x)

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
    version='1.0.0',
)

app.add_middleware(CORSMiddleware, allow_origins=['*'])

app.add_exception_handler(Exception, python_exception_handler)


@app.on_event('startup')
async def startup_event():
    """Initialize FastAPI and variables"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Pytorch using device: {device}')

    model = ResModel().to(device)
    model.load_state_dict(torch.load('model_chp/ResModel.pth', map_location=device))
    model.eval()

    segment_path = 'model_chp/sam_vit_l_0b3195.pth'

    app.package = {
        'model': model,
        'device': device,
        'segment_model_path': segment_path
    }


@app.get("/")
def home():
    return "API is working as expected"


@app.get("/about")
def show_about():
    """Get deployment information, for debugging."""

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
    }


@app.post('/predict')
async def do_prediction(model: Model, file: UploadFile = File(...)) -> dict:
    """Performs prediction on an uploaded image using a specified model

        :param model: The model to use for prediction.
        :param file: The uploaded image file.
        :returns: The dictionary containing the prediction and confidence.
        """

    filename = file.filename
    file_extension = filename.split('.')[-1] in ('jpg', 'png')
    if not file_extension:
        raise HTTPException(status_code=415, detail='Unsupported file provided.')

    # Reads the content of the uploaded file
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    image.save(f'images_uploaded/{filename}')

    y = predict(app.package, image)[0]

    probabilities = np.exp(y) / np.sum(np.exp(y))

    # Get the index of the predicted class with the highest probability
    predicted_class_index = np.argmax(probabilities)

    confidence_percentage = round(probabilities[predicted_class_index] * 100, 4)

    pred = class_names[predicted_class_index]

    return {
        'prediction': pred,
        'confidence': confidence_percentage
    }
