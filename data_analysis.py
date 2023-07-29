import os
import re
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import Image


def get_class_dir_names(root_path: str) -> List[str]:
    """Traverse the root dataset directory for the names of its subdirectories and returns a list of their names.

    :param root_path: The root directory of dataset.
    :returns: A list of names of the subdirectories of the root directory.
    """

    return [class_dir.name for class_dir in os.scandir(root_path) if class_dir.is_dir()]


def image_to_ndarray(image_path: str) -> np.ndarray[int]:
    """Load the image from the specified path and returns it as a NumPy array.

    :param image_path: The path to the image file
    :returns: A NumPy array representation of the image
    """

    img = Image.open(image_path)
    return np.array(img)


def plot_images(root_path: str) -> None:
    """Plot a 4x4 grid of images from the dataset, with each image corresponding
    to a subdirectory of the root directory.

    :param root_path: The root directory of the dataset.
    """

    # Get the names of the subdirectories in the root directory
    class_dirs_names = get_class_dir_names(root_path)

    # Loop over the subdirectories and plot the first image from each directory, nrows x ncols >= 34 for all classes
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 20))
    fig.suptitle('Images in height x width shape', fontsize=16, fontweight='bold')

    for i, class_dir_name in enumerate(class_dirs_names):
        # Get a list of all the files in the subdirectory and the path of the first image
        files_list = os.listdir(os.path.join(root_path, class_dir_name))
        image_path = os.path.join(root_path, class_dir_name, files_list[0])

        image = image_to_ndarray(image_path)
        height, width = image.shape[:2]

        # Plot the image on the subplot at row i//4, column i%4
        if i < 16:
            axes[i // 4, i % 4].imshow(image)
            axes[i // 4, i % 4].set_title(f'{class_dir_name}\nDimensions: {height}x{width}')
            axes[i // 4, i % 4].axis('off')
    plt.show()


def print_dataset_info(root_path: str) -> None:
    """Print information about the classes in the dataset, including the unique plants and diseases,
     and the number of classes, unique plants, unique diseases, and shared diseases.

    :param root_path: The root directory of the dataset.
    """

    # Get the names of the subdirectories of the root directory
    class_dirs_names = get_class_dir_names(root_path)

    # Create sets of unique plants and diseases
    unique_plants = set(name.split('_')[0] for name in class_dirs_names)
    unique_diseases = set(' '.join(name.split('_')[1:]) for name in class_dirs_names if not re.search('healthy', name))

    print(f'Unique plants are: \n{unique_plants}\n')
    print(f'Unique disease are: \n{unique_diseases}\n')

    dict_class_numbers = {
        'Number of classes': len(class_dirs_names),
        'Unique plants': len(unique_plants),
        'Unique diseases': len(unique_diseases),
        'Number of shared diseases': len(class_dirs_names) - len(unique_plants) - len(unique_diseases),
    }

    print(dict_class_numbers)


def get_num_images_per_class(root_path: str) -> pd.DataFrame:
    """Return a Pandas DataFrame containing the number of images in each subdirectory (i.e., class)
    of the root directory.

    :param root_path: The root directory of the dataset.
    :returns: A pandas DataFrame containing number of images in each subdirectory.
    """

    class_name_num_images_list = []
    for subdir in os.scandir(root_path):
        num_images = 0
        if subdir.is_dir():
            for filename in os.listdir(subdir):
                num_images += 1

            # Add the class name and number of images to the list
            class_name = subdir.name
            class_name_num_images_dict = {'class_name': class_name, 'number_of_images': num_images}
            class_name_num_images_list.append(class_name_num_images_dict)

    df = pd.DataFrame(class_name_num_images_list)

    return df


def plot_images_distribution(df: pd.DataFrame) -> None:
    """Plot a horizontal bar chart showing the distribution of images across classes.

    :param df: A Pandas DataFrame containing the number of images in each subdirectory of the root directory.
    """

    df_sorted = df.sort_values('number_of_images')

    plt.figure(figsize=(20, 15))
    plt.barh(df_sorted['class_name'], df_sorted['number_of_images'])
    plt.title("Distribution of data")
    plt.xlabel("Number of images")
    plt.ylabel("Class names")
    plt.show()


if __name__ == '__main__':
    root_dir = 'data/plant_dataset_original/plant_diseases_images'

    print_dataset_info(root_dir)

    plot_images(root_dir)

    df = get_num_images_per_class(root_dir)
    df_sorted = df.sort_values('number_of_images')
    print(df_sorted, '\n')

    plot_images_distribution(df)

    print(f"Total number of images: {df['number_of_images'].sum()}")
