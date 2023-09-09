import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from app.util.custom_clahe import CustomCLAHE
from PIL import Image
from torch import Tensor


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
