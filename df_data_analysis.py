import os
import re
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from PIL import Image


def get_class_dirs(root_path: str) -> List[str]:
    return [class_dir.name for class_dir in os.scandir(root_path) if class_dir.is_dir()]


def get_image(image_path: str) -> np.ndarray[int]:
    img = Image.open(image_path)
    return np.array(img)


def plot_images(root_path: str) -> None:
    class_dirs_names = get_class_dirs(root_path)
    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 20))

    for i, class_dir_name in enumerate(class_dirs_names):
        files_list = os.listdir(os.path.join(root_path, class_dir_name))
        image_path = os.path.join(root_path, class_dir_name, files_list[0])
        image = get_image(image_path)
        width, height = image.shape[:2]

        if i < 16:
            axes[i // 4, i % 4].imshow(image)
            axes[i // 4, i % 4].set_title(f'{class_dir_name}\nDimensions: {width}x{height}')
            axes[i // 4, i % 4].axis('off')
    plt.show()


def print_class_distribution(root_path: str) -> None:
    class_dirs_names = get_class_dirs(root_path)

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


if __name__ == '__main__':
    root_dir = 'data/plant_dataset_original/plant_diseases_images'
    plot_images(root_dir)

    class_name_num_images_list = []
    for subdir in os.scandir(root_dir):
        num_images = 0
        if subdir.is_dir():
            for filename in os.listdir(subdir):
                img_path = os.path.join(subdir, filename)
                num_images += 1

            class_name = subdir.name
            class_name_num_images_dict = {'class_name': class_name, 'number_of_images': num_images}
            class_name_num_images_list.append(class_name_num_images_dict)

    df = pd.DataFrame(class_name_num_images_list)
    df_sorted = df.sort_values('number_of_images')
    print(df_sorted, '\n')

    plt.figure(figsize=(20, 15))
    plt.barh(df_sorted['class_name'], df_sorted['number_of_images'])
    plt.title("Distribution of data")
    plt.xlabel("Number of images")
    plt.ylabel("Class names")
    plt.show()

    print_class_distribution(root_dir)
