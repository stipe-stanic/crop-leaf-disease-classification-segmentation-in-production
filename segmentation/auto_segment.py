import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as F

from typing import List, Tuple
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torch import Tensor
from leaf_of_interest_algo import leaf_of_interest_selection


def get_device() -> torch.device:
    """Get the device for running PyTorch computations.

    :returns: torch.device: The selected device (CPU or GPU).
    """

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    return device


def show_single_image(image: np.ndarray | Tensor, figsize: Tuple[int, int] = (10, 10), axis: bool = True) -> None:
    """Plot a single image.

    :param image: The input image, should be in RGB color space.
    :param figsize: Size of the figure.
    :param axis: Boolean flag to show coordinates.
    """

    if isinstance(image, Tensor):
        image = image.detach()
        image = F.to_pil_image(image)
        image = np.asarray(image)

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis(axis)
    plt.show()


def show_anns(anns: List[dict], ax) -> None:
    """Display annotations on the current axis as colored masks.

    :param anns: A list of dictionaries representing annotations.
                Each dictionary should contain a 'segmentation' key, which is a binary mask (2D numpy array)
                and other necessary information for displaying the annotations.
    """

    if len(anns) == 0:
        return

    # Sort the annotations based on area in descending order
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    # Disable autoscaling of the axis to ensure annotations are displayed correctly
    ax.set_autoscale_on(False)

    # Create a blank image with an alpha channel (4th channel) for transparency
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # random color with transparency
        img[m] = color_mask

    # Overlay the colored masks on top of the original image
    ax.imshow(img)


def display_annotations(image: np.ndarray, anns: List[dict], axis: bool = True):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(anns, plt.gca())
    plt.axis(axis)
    plt.show()


def show_point_grid(point_grid: List[np.ndarray], image: np.ndarray, ax, marker_size: int = 375) -> None:
    """Display an image with scatter plot overlay of points from the given grid.

    :param point_grid: A list of NumPy arrays containing the points to be displayed on the image.
    :param image: The image to be displayed
    :param ax: The matplotlib axis on which the image and scatter plot will be drawn.
    :param marker_size: Size of the markers in the scatter plot.
    """

    # Extract the grid of points from the list (assuming there's only one set of points in the list)
    grid = point_grid[0]
    height, width = image.shape[0], image.shape[1]

    # Display the image on the given axis
    ax.imshow(image)

    # Points are normalized between [0, 1] so we multiply by height and width to get actual coords
    ax.scatter(grid[:, 0] * width, grid[:, 1] * height, color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


if __name__ == '__main__':
    #image = cv2.imread("./5505139.jpg")
    image = cv2.imread("blight_corn_9.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    show_single_image(image, axis=False)

    sam_checkpoint = "../models_storage/pretrained/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = get_device()
    print(device)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # points to specified location in the image (x, y)
    point_grid = [np.array([
        [0.5, 0.25], [0.25, 0.25], [0.75, 0.25],  # top (left, center, right)
        [0.5, 0.5], [0.25, 0.5], [0.75, 0.5],  # middle (left, center, right)
        [0.5, 0.75], [0.25, 0.75], [0.75, 0.75],  # bottom (left, center, right)
        ])]

    fig, ax = plt.subplots(figsize=(10, 10))
    show_point_grid(point_grid, image, ax)
    plt.show()

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=None, point_grids=point_grid)
    masks = mask_generator.generate(image)
    print(len(masks))
    display_annotations(image, masks)

    leaf_to_segment = leaf_of_interest_selection(masks, image)
    display_annotations(image, [leaf_to_segment])

    segmented_image = cv2.bitwise_and(image, image, mask=leaf_to_segment['segmentation'].astype(np.uint8))
    show_single_image(segmented_image, axis=False)
