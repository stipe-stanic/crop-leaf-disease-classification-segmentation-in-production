import os
import re

import cv2
import numpy as np
import random

from typing import Tuple, List
from PIL import Image
from img_aug_transform import ImgAugGenerate


def rename_subdir(root_dir: str) -> None:

    """Renames subdirectories to standard naming convention: plant_*_disease_*

    :param root_dir: The root directory of the dataset
    """

    for dir_name in os.listdir(root_dir):
        # Remove repetition of any word
        dir_name_target = ' '.join(sorted(set(dir_name.lower().split()), key=dir_name.lower().split().index))

        # Remove list of words
        words_to_remove = ["leaf"]
        for word in words_to_remove:
            dir_name_target = dir_name_target.replace(f'_{word}_', '_')

        # Replace spaces with underscores
        dir_name_target = dir_name_target.replace(' ', '_')

        # Rename the directory
        os.rename(os.path.join(root_dir, dir_name), os.path.join(root_dir, dir_name_target))


def print_subdir_name(root_dir: str) -> None:
    """Prints a list of subdirectories' names

    :param root_dir: The root directory of the dataset
    """

    subdirs = [os.path.basename(x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
    print(subdirs)


def get_mean_std_of_pixel_values(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and standard deviation of pixel values in the dataset.


    :param root_dir: :param root_dir: The root directory of the dataset
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
        if class_dir.is_dir() and re.search('apple', class_dir.name):
            # Limits the number of images per class to reduce processing time
            for filename in os.listdir(class_dir):
                if i > 500:
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
