import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import torch
import seaborn as sns

import cv2
import numpy as np
import random

from torchvision.datasets.folder import DatasetFolder
from torch.utils.data import DataLoader, Subset
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from typing import Tuple, List
from PIL import Image
from img_aug_generate import ImgAugGenerate
from train import config
from segmentation.leaf_of_interest_algo import leaf_of_interest_selection
from segmentation.segment_util import prepare_mask_to_crop, crop_segmented_image


def rename_subdirs(root_dir: str) -> None:

    """Renames subdirectories to standard naming convention: plant_*_disease_*

    :param root_dir: The root directory of the dataset.
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
    """Rename files within a subdirectory based on certain rules.

    :param root_dir: The root directory of the dataset.
    :param sub_dir_name: The name of the subdirectory within the root directory.
    """

    sub_dir = os.path.join(root_dir, sub_dir_name)
    for file_name in os.listdir(sub_dir):
        # Splits the original file name into words, removes duplicates,
        # and sorts the words based on their original order or appearance
        file_name_target = ' '.join(sorted(set(file_name.lower().split()), key=file_name.lower().split().index))

        words_to_remove = ['leaf', '(including_sour)', '(maize)', 'esca', 'flip',
                           'and', 'two', 'spotted']

        # Removes specific words from the file name
        for word in words_to_remove:
            if word in file_name_target:
                file_name_target = file_name_target.replace(f'_{word}_', '_')
                file_name_target = file_name_target.replace(f'_{word}', '')

        file_name_target = file_name_target.replace(' ', '_')

        # Removes first occurrence of ".jpg", if there are multiple occurrences
        if file_name_target.count('.jpg') > 1:
            file_name_target = file_name_target.replace('.jpg', '', 1)

        os.rename(os.path.join(sub_dir, file_name), os.path.join(sub_dir, file_name_target))


def print_subdir_names(root_dir: str) -> None:
    """Print a list of subdirectories' names

    :param root_dir: The root directory of the dataset.
    """

    subdirs = [os.path.basename(x) for x in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, x))]
    print(subdirs)


def resize_image(img, target_size: Tuple[int, int]) -> Image:
    """Resize the image while maintaining the aspect ratio and pad or crop it to the target size.

    :param img: The input image.
    :param target_size: The desired size (tuple of width and height).
    :returns: The resized and processed image.
    """
    width, height = img.size
    target_width, target_height = target_size

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Determine the resizing dimensions while maintaining the aspect ratio
    if aspect_ratio > 1:
        # Landscape orientation
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Portrait or square orientation
        new_width = int(target_height * aspect_ratio)
        new_height = target_height

    # Resize the image while maintaining the aspect ratio
    resized_img = img.resize((new_width, new_height))

    # Pad or crop the image to the target size
    padded_img = Image.new("RGB", target_size)
    padded_img.paste(resized_img, ((target_width - new_width) // 2, (target_height - new_height) // 2))

    return padded_img


def get_mean_std_of_pixel_values(train_subset: Subset, dataset: DatasetFolder) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and standard deviation of pixel values in the training dataset.

   :param train_subset: Subset of the data.
   :param dataset: The dataset containing images.
   :returns: Tuple containing the mean and standard deviation of pixel values as NumPy arrays.
   :raises: ValueError if no images were found in the dataset.
   """

    # Initializes variables for accumulating mean and std
    n_images = len(train_subset)
    if n_images == 0:
        raise ValueError("No images found in the dataset")

    total_mean = np.zeros((3, ))
    total_std = np.zeros((3, ))

    for image_idx in train_subset.indices:
        img, _ = dataset[image_idx]

        # Gets pixel values as a list
        img_array = np.array(img).reshape((224, 224, 3))

        # Accumulates mean value and standard deviation for each channel
        mean = np.mean(img_array, axis=(0, 1))
        std = np.std(img_array, axis=(0, 1))

        total_mean += mean
        total_std += std

    mean_pixel_values = total_mean / n_images
    std_pixel_values = total_std / n_images

    return mean_pixel_values, std_pixel_values


def print_class_distribution(
        dataset: DatasetFolder, train_dataset: Subset,
        val_dataset: Subset, test_dataset: Subset) -> None:
    """Print the distribution of classes in the train, validation, and test datasets.

   :param dataset: The original dataset.
   :param train_dataset: Subset of the dataset used for training.
   :param val_dataset: Subset of the dataset used for validation.
   :param test_dataset: Subset of the dataset used for testing.
   """

    # Creates dataloaders for train, validation and test datasets
    train_target_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    val_target_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_target_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Collects target labels from train, validation and test datasets
    train_targets = []
    for batch in train_target_loader:
        train_targets.extend(batch[1].tolist())

    val_targets = []
    for batch in val_target_loader:
        val_targets.extend(batch[1].tolist())

    test_targets = []
    for batch in test_target_loader:
        test_targets.extend(batch[1].tolist())

    # Calculates class counts for each subset of the dataset
    train_counts = [train_targets.count(i) for i in range(len(dataset.classes))]
    val_counts = [val_targets.count(i) for i in range(len(dataset.classes))]
    test_counts = [test_targets.count(i) for i in range(len(dataset.classes))]

    print("Train class distribution:", train_counts)
    print("Val class distribution:", val_counts)
    print("Test class distribution:", test_counts)


def trim_num_images(root_dir: str, classes_to_trim: List[str], trim_number: int) -> None:
    """Randomly removes images from the dataset so no class has over n number of images

    :param root_dir: The root directory of the dataset
    :param classes_to_trim: List of class names to trim
    :param trim_number: Maximum number of images to retain per class
    """

    for class_dir_name in classes_to_trim:
        class_dir = os.path.join(root_dir, class_dir_name)
        files_list = os.listdir(class_dir)

        if len(files_list) > trim_number:
            num_files_to_remove = len(files_list) - trim_number

            # Randomly selects images from the class's directory to remove
            files_to_remove = random.sample(files_list, num_files_to_remove)

            for file_name in files_to_remove:
                file_path = os.path.join(class_dir, file_name)
                os.remove(file_path)


def generate_aug_images(root_path: str, class_name: str, num_images: int) -> None:
    """Generate augmented images for a given class from the images located in the specified root directory.

    :param root_path: Root directory path of the dataset.
    :param class_name: Name of the class to generate augmented images for.
    :param num_images: Number of augmented images to generate.
    """

    class_dir = os.path.join(root_path, class_name)
    image_files = os.listdir(class_dir)

    # Class for image augmentation
    augment = ImgAugGenerate()

    for i in range(num_images):
        random_image_file = random.choice(image_files)
        image_path = os.path.join(class_dir, random_image_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Applies augmentation using __call__ of instance
        augmented_image = augment(image)

        image_name = f'{class_name}_augmented_{i+1}.jpg'
        save_file_path = os.path.join(class_dir, image_name)

        # Converts the augmented image to PIL Image
        augmented_image = Image.fromarray(augmented_image)
        augmented_image.save(save_file_path)


def plot_rgb_channel_distribution(root_dir: str) -> None:
    """Plot the distribution of RGB channel values in a set of images.

    :param root_dir: Root directory of the dataset.
    """

    # Stores the mean channel values for each image
    rgb_values = []

    for class_dir in os.scandir(root_dir):
        i = 0
        if class_dir.is_dir():
            # Limits the number of images per class to reduce processing time
            for filename in os.listdir(class_dir):
                if i > 100:
                    break

                img_path = os.path.join(class_dir, filename)
                img = Image.open(img_path)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Converts an image to a numpy array
                pixels = np.array(img)

                # Calculates mean value of each channel separately along the first and second axes of the array which
                # correspond to height and width of the image, shape=(3,)
                channel_means = np.mean(pixels, axis=(0, 1))

                rgb_values.append(channel_means)
                i += 1

    rgb_values = np.array(rgb_values)

    df = pd.DataFrame(rgb_values, columns=['Red', 'Green', 'Blue'])
    colors = ['red', 'green', 'blue']

    # Plots boxplot for the channel distributions
    sns.boxplot(data=df, palette=colors)
    plt.title("Distribution of channels: red, green, blue")
    plt.xlabel('Channel')
    plt.ylabel('Pixel values')
    plt.show()


def segment_images(root_dir: str, root_dir_segment: str, model_checkpoint: str) -> None:
    """Segment leaves using the provided model checkpoint and save the cropped images.

    :param root_dir: Root directory to save the segmented images
    :param root_dir_segment: Root directory containing images to be segmented
    :param model_checkpoint: Path to the model checkpoint for segmentation
    """

    sam_checkpoint = model_checkpoint
    model_type = sam_checkpoint.split('/')[-1][4:9]  # example: 'vit_l'

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    for class_dir in os.scandir(root_dir_segment):
        if class_dir.is_dir():
            for i, filename in enumerate(np.asarray(os.listdir(class_dir))):
                img_path = os.path.join(class_dir, filename)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                img_height, img_width = image.shape[0], image.shape[1]

                # points to specified location in the image (x, y)
                point_grid = [np.array([
                    [0.25, 0.25], [0.5, 0.25], [0.75, 0.25],  # top (left, center, right)
                    [0.25, 0.5], [0.5, 0.5], [0.75, 0.5],  # middle (left, center, right)
                    [0.25, 0.75], [0.5, 0.75], [0.75, 0.75],  # bottom (left, center, right)
                ])]

                min_mask_area = 0.05 * (img_height * img_width)
                mask_generator = SamAutomaticMaskGenerator(sam,
                                                           points_per_side=None,
                                                           point_grids=point_grid,
                                                           stability_score_thresh=0.95,
                                                           min_mask_region_area=min_mask_area
                                                           )
                masks = mask_generator.generate(image)

                # Skip processing if no masks are found
                if len(masks) == 0:
                    continue

                leaf_to_segment = leaf_of_interest_selection(masks, image)
                segmented_image = cv2.bitwise_and(image, image, mask=leaf_to_segment['segmentation'].astype(np.uint8))

                boxes, segmented_image = prepare_mask_to_crop(leaf_to_segment['segmentation'], segmented_image.copy())

                # Skip processing if no bounding box is found
                if boxes.shape[0] == 0:
                    continue

                cropped_image = crop_segmented_image(boxes, segmented_image)

                save_path = os.path.join(root_dir, class_dir.name, f'{class_dir.name}_segmented_{i + 1}.jpg')
                plt.imsave(save_path, cropped_image.numpy().transpose(1, 2, 0))
