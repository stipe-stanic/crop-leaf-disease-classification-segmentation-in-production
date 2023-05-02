import os

from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import torchvision.transforms

from torchvision.datasets.folder import default_loader, DatasetFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader

def valid_file(filename: str) -> bool:
    """Check if current file has valid extension"""

    return filename.lower().endswith(('.jpg', '.png'))


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    This function searches for subdirectories in the specified directory and returns a list of class names and
    dictionary mapping each class name to its corresponding index.

    The function only includes subdirectories whose names match the regular expression 'apple'. This is intended to
    filter the classes for a specific type of dataset.

    :param directory: The root directory of the training dataset.

    :returns: A tuple containing a list of strings where each string is the name of a class folder and dictionary
    that maps each class name to its corresponding index.

    :raises FileNotFoundError: If no class folders are found in the specified directory.
    """

    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir()
                     and re.search('apple', entry.name))
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class CustomImageFolder(torchvision.datasets.DatasetFolder):
    """ Implements custom ImageFolder class that overrides DatasetFolder methods, so it's possible to load only
    specific subdirectories(classes) of the directory instead of the whole directory.

    Enables two valid extensions (.jpg, .png)
    """

    def __init__(self, root, transform=None, target_transform=None, loader=default_loader, is_valid_file=valid_file):
        super().__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return find_classes(directory)


def show_dataset(dataset: DatasetFolder, n=6) -> None:
    """Shows grid of images as a single image

    :param dataset: Loaded torchvision dataset
    :param n: Number of rows and columns
    """

    # Transform image from tensor to PILImage
    transform = torchvision.transforms.ToPILImage()
    img = np.vstack([np.hstack([np.asarray(transform(dataset[i][0])) for _ in range(n)])
                     for i in range(n)])
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def show_images(dataset_loader: DataLoader , num_of_images: int) -> None:
    """Displays images before feeding them to the model

    :raises AssertionError: If number of images to display exceeds batch size
    """

    batch_size = dataset_loader.batch_size

    try:
        assert num_of_images < batch_size,\
            f"Number of images to display exceeds batch size: {num_of_images} > {batch_size}"

        data_iter = iter(dataset_loader)
        images, labels = next(data_iter)

        for index in range(1, num_of_images + 1):
            plt.subplot(2, 10, index)
            plt.axis('off')
            plt.title(str(labels.numpy()[index]))

            # imshow function expects an image with the shape (height, width, channels)
            img = images[index].numpy().transpose((1, 2, 0)).squeeze()
            # img = (img - np.min(img)) / (np.max(img) - np.min(img))
            plt.imshow(img)
        plt.show()
    except AssertionError as msg:
        print("Error:", msg)


def loader_shape(dataset_loader: DataLoader) -> Tuple[torch.Size, torch.Size]:
    """Prints shape of loaded dataset

    :returns: Tuple of tensor shapes (images, labels)
    """

    data_iter = iter(dataset_loader)
    images, labels = next(data_iter)

    return images.shape, labels.shape


def train():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(device)

    root_dir = "data/plant_dataset_original/plant_diseases_images"

    dataset_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    dataset = CustomImageFolder(root=root_dir, loader=default_loader, transform=dataset_transforms)
    show_dataset(dataset)

    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(10, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.RandomGrayscale(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset.dataset.transform = train_transforms
    show_dataset(train_dataset)

    batch_size = 32

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                             num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=2, pin_memory=True)

    images_shape, labels_shape = loader_shape(train_loader)
    print(f'Images shape: {images_shape}\nLabels shape: {labels_shape}')

    show_images(train_loader, num_of_images=20)

if __name__ == '__main__':
    train()