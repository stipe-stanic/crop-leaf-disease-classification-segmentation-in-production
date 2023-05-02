import os
import re
import numpy as np

from typing import Tuple
from PIL import Image


def rename_subdir(root_dir: str) -> None:
    """Renames subdirectories to standard naming convention: plant_*_disease_*"""

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
    """Prints a list of subdirectories' names"""

    subdirs = [os.path.basename(x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
    print(subdirs)


def get_mean_std_of_pixel_values(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    # Initialize variables for accumulating mean and std
    n_images = 0
    total_mean = np.zeros((3, ))
    total_std = np.zeros((3, ))

    # Iterate over images in directory
    for class_dir in os.scandir(root_dir):
        i = 0
        if class_dir.is_dir() and re.search('apple', class_dir.name):
            for filename in os.listdir(class_dir):
                if i > 500:
                    break
                img_path = os.path.join(class_dir, filename)
                img = Image.open(img_path)
                pixel_list = list(img.getdata())
                img_array  = np.array(pixel_list).reshape((img.size[1], img.size[0], 3))
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

