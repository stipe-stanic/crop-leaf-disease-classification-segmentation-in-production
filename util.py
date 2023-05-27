import os
import re
import torch

import cv2
import numpy as np
import random

from torchvision.datasets.folder import DatasetFolder
from torch.utils.data import DataLoader, Subset
from typing import Tuple, List
from PIL import Image
from img_aug_transform import ImgAugGenerate
from train import config


def rename_subdir(root_dir: str) -> None:

    """Renames subdirectories to standard naming convention: plant_*_disease_*

    :param root_dir: The root directory of the dataset
    """

    for dir_name in os.listdir(root_dir):
        # Remove repetition of any word
        dir_name_target = ' '.join(sorted(set(dir_name.lower().split()), key=dir_name.lower().split().index))

        # Remove list of words
        words_to_remove = ['leaf']
        for word in words_to_remove:
            dir_name_target = dir_name_target.replace(f'_{word}_', '_')

        # Replace spaces with underscores
        dir_name_target = dir_name_target.replace(' ', '_')

        # Rename the directory
        os.rename(os.path.join(root_dir, dir_name), os.path.join(root_dir, dir_name_target))


def rename_subdir_files(root_dir: str, sub_dir_name: str) -> None:
    sub_dir = os.path.join(root_dir, sub_dir_name)
    for file_name in os.listdir(sub_dir):
        file_name_target = ' '.join(sorted(set(file_name.lower().split()), key=file_name.lower().split().index))

        words_to_remove = ['leaf', '(including_sour)', 'in', '(maize)', 'esca', 'flip',
                           'and', 'two', 'spotted', 'mite']
        for word in words_to_remove:
            if word in file_name_target:
                file_name_target = file_name_target.replace(f'_{word}_', '_')
                file_name_target = file_name_target.replace(f'_{word}', '')

        file_name_target = file_name_target.replace(' ', '_')

        # Remove first occurrence of ".jpg", since some files have double appearance
        if file_name_target.count('.jpg') > 1:
            file_name_target = file_name_target.replace('.jpg', '', 1)

        if sub_dir_name == 'tomato_mosaic_virus' and file_name_target.count('tomato') > 1:
            file_name_target = file_name_target.replace(f'_tomato_', '_', 1)

        os.rename(os.path.join(sub_dir, file_name), os.path.join(sub_dir, file_name_target))


def print_subdir_name(root_dir: str) -> None:
    """Prints a list of subdirectories' names

    :param root_dir: The root directory of the dataset
    """

    subdirs = [os.path.basename(x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
    print(subdirs)


def get_mean_std_of_pixel_values(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and standard deviation of pixel values in the dataset.

    :param root_dir: The root directory of the dataset
    :returns: Tuple containing the mean and standard deviation of pixel values as NumPy arrays.
    :raises: ValueError if no images were found in the dataset
    """

    # Initialize variables for accumulating mean and std
    n_images = 0
    total_mean = np.zeros((3, ))
    total_std = np.zeros((3, ))

    # Iterate over images in directory
    for class_dir in os.scandir(root_dir):
        i = 0
        if class_dir.is_dir():
            # Limits the number of images per class to reduce processing time
            for filename in os.listdir(class_dir):
                if i > 100:
                    break
                img_path = os.path.join(class_dir, filename)
                img = Image.open(img_path)
                pixel_list = list(img.getdata())
                img_array = np.array(pixel_list).reshape((img.size[1], img.size[0], 3))
                img_array = img_array.astype(np.float32) / 255.0

                mean = np.mean(img_array, axis=(0, 1))
                std = np.std(img_array, axis=(0, 1))

                total_mean += mean
                total_std += std
                n_images += 1
                i += 1

    if n_images > 0:
        mean_pixel_values = total_mean / n_images
        std_pixel_values = total_std / n_images
    else:
        raise ValueError("No images found in dataset")

    return mean_pixel_values, std_pixel_values


def print_class_distribution(
        dataset: DatasetFolder, train_dataset: Subset,
        val_dataset: Subset, test_dataset: Subset) -> None:
    # Check distribution of classes in train, val and test datasets
    train_target_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_target_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_target_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    train_targets = []
    for batch in train_target_loader:
        train_targets.extend(batch[1].tolist())

    val_targets = []
    for batch in val_target_loader:
        val_targets.extend(batch[1].tolist())

    test_targets = []
    for batch in test_target_loader:
        test_targets.extend(batch[1].tolist())

    train_counts = [train_targets.count(i) for i in range(len(dataset.classes))]
    val_counts = [val_targets.count(i) for i in range(len(dataset.classes))]
    test_counts = [test_targets.count(i) for i in range(len(dataset.classes))]

    print("Train class distribution:", train_counts)
    print("Val class distribution:", val_counts)
    print("Test class distribution:", test_counts)


def trim_num_images(root_path: str, classes_to_trim: List[str], trim_number: int) -> None:
    """Randomly removes images from the dataset so no class has over n number of images

    :param root_path: The root directory of the dataset
    :param classes_to_trim: List of class names to trim
    :param trim_number: Maximum number of images to retain per class
    """

    for class_dir_name in classes_to_trim:
        class_dir = os.path.join(root_path, class_dir_name)
        files_list = os.listdir(class_dir)

        if len(files_list) > trim_number:
            num_files_to_remove = len(files_list) - trim_number
            files_to_remove = random.sample(files_list, num_files_to_remove)

            for file_name in files_to_remove:
                file_path = os.path.join(class_dir, file_name)
                os.remove(file_path)


def generate_aug_images(root_path: str, class_name: str, num_images: int) -> None:
    image_dir = os.path.join(root_path, class_name)
    image_files = os.listdir(image_dir)

    augment = ImgAugGenerate()

    for i in range(num_images):
        random_image_file = random.choice(image_files)
        image_path = os.path.join(image_dir, random_image_file)

        image = cv2.imread(image_path)
        augmented_image = augment(image)

        image_name = f'{class_name}_augmented_{i+1}.jpg'
        save_file_path = os.path.join(image_dir, image_name)
        augmented_image = Image.fromarray(augmented_image)
        augmented_image.save(save_file_path)
