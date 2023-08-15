import io
import os
import sys
import matplotlib.pyplot as plt
import torchvision
import uvicorn
import numpy as np
import torch
import deployment.local.config as local_config

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from PIL import Image
from train.res_model import ResModel
from torch import Tensor
from local_exception_handler import python_exception_handler
from custom_clahe import CustomCLAHE


def preprocess(input_image: Image) -> Tensor:
    """Preprocess an input image using a provided transform and return a tensor.

    :param input_image: The input image to be preprocessed.
    :returns: The preprocessed image tensor.
    """

    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        CustomCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),  # increases contrast in a smart way
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
    ])

    img = dataset_transforms(input_image)
    print(img.shape)

    # Converts the transformed image back to PIL Image object
    transform_back = torchvision.transforms.ToPILImage()

    # Converts the image back to numpy array
    image_processed = np.asarray(transform_back(img))
    plt.imshow(image_processed)
    plt.axis('off')
    plt.show()

    # Adds an extra dimension
    img_tensor = torch.unsqueeze(img, 0)

    return img_tensor


def predict(package: dict, input: Image) -> np.ndarray:
    """Run prediction using a provided model on a preprocessed input image.

    :param package: A dictionary containing the 'model', 'device', and 'transform' objects.
    :param input: The input image to be classified.
    :returns: The prediction values for each class as a numpy array.
    """

    x = preprocess(input)

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
    model.load_state_dict(torch.load('../../../models_storage/export_models/ResModel.pth', map_location=device))
    model.eval()

    app.package = {
        'model': model,
        'device': device
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

    plt.imshow(image)
    plt.axis('off')
    plt.show()

    y = predict(app.package, image)[0]

    probabilities = np.exp(y) / np.sum(np.exp(y))

    # Get the index of the predicted class with the highest probability
    predicted_class_index = np.argmax(probabilities)

    confidence_percentage = round(probabilities[predicted_class_index] * 100, 4)

    pred = local_config.class_names[predicted_class_index]
    print(f'Predicted: {pred}\nConfidence: {confidence_percentage}')

    return {
        'prediction': pred,
        'confidence': confidence_percentage
        }


if __name__ == '__main__':
    uvicorn.run("main:app", host=local_config.host, port=local_config.port,
                log_config="local_log.ini", reload=True, timeout_keep_alive=60)
