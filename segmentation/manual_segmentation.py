import os

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

from typing import Tuple, List
from segment_anything import SamPredictor, sam_model_registry
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import masks_to_boxes
from torch import Tensor


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


def show_points(coords, labels, ax, marker_size=375) -> None:
    """Visualize points in a 2D space based on their labels.

    :param coords: 2D array representing the coordinates of points in a 2D space.
    :param labels: 1D array representing the labels assigned to each point in 'coords', binary values (0 or 1) indicate
    the category or class of each point.
    :param ax: Axes instance where the scatter plot will be drawn.
    :param marker_size: Size of the marker used to plot the points.
    """

    # Filter points with label 1 (positive category) and label 0 (negative category)
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)  # [:, 0] all rows, first column
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask: np.ndarray, ax, random_color: bool = False) -> None:
    """Visualize binary mask

    :param mask: 2D binary array representing the mask to be visualized.
    :param ax: Axes instance where the scatter plot will be drawn.
    :param random_color: A boolean flag indicating whether to use a random color for the mask overlay.
    :return:
    """

    if random_color:
        # Uses a random RGB color with alpha (opacity) value of 0.6
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Uses a fixed RGB color (light blue) with alpha (opacity) value of 0.6
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    # Reshapes the mask and color arrays to match the dimensions of the mask
    h, w = mask.shape[-2:]

    # The third dimension with size 1 is added to create a single-channel image-like representation of the mask
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Overlays the mask on the specified axes
    ax.imshow(mask_image)


def plot_masks(masks: np.ndarray, scores: np.ndarray, image: np.ndarray) -> None:
    """Plot multiple masks with their associated scores on separate figures.

    :param masks: 3D array representing a collection of binary masks to be visualized, each mask is 2D binary array.
    :param scores: 1D array representing the scores associated with each maks.
    :param image: Original input image.
    """

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def show_drawn_masks_to_crop(images: Tensor | List[Tensor]) -> None:
    """Display one or more tensor images with segmentation masks drawn on top.

    :param images: Input tensor image.
    """

    if not isinstance(images, list):
        images = [images]

    fix, axs = plt.subplots(ncols=len(images), squeeze=False)
    for i, img in enumerate(images):
        # If the image is a tensor, detaches it to remove any computation graph bindings
        img = img.detach()
        img = F.to_pil_image(img)

        # Displays the image in the corresponding subplot
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


if __name__ == '__main__':
    image = cv2.imread("../deployment/local/client/images_post/corn/blight_corn_451 - Copy.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    show_single_image(image)

    sam_checkpoint = "../models_storage/pretrained/sam_vit_l_0b3195.pth"
    model_type = sam_checkpoint.split('/')[-1][4:9]  # example: 'vit_l'

    device = get_device()
    print(device)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # maps to a function
    sam = sam.to(device)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    input_point = np.array([[400, 600]])  # points to specified location in the image (x, y)
    input_label = np.array([1])  # 1: foreground point, 0: background point

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    print(masks.shape)  # (number_of_masks) x H x W

    plot_masks(masks, scores, image)

    if masks.size > 1:
        mask_to_crop = masks[np.argmax(scores)].astype(np.uint8)
    else:
        mask_to_crop = masks.copy().astype(np.uint8)

    segmented_image = cv2.bitwise_and(image, image, mask=mask_to_crop)
    show_single_image(segmented_image, axis=False)

    mask_to_crop = np.squeeze(mask_to_crop)

    # Pixels can either be in foreground or in background.
    obj_ids = np.unique(mask_to_crop)

    # Removing background pixels
    obj_ids = obj_ids[1:]

    # Splits the color-encoded mask into a set of boolean masks.
    to_crop_mask = mask_to_crop == obj_ids[:, None, None]

    segmented_image = np.transpose(segmented_image, (2, 0, 1))  # (channels x height x width)

    to_crop_mask = torch.from_numpy(to_crop_mask)
    segmented_image = torch.from_numpy(segmented_image)

    # drawn_masks = draw_segmentation_masks(image, to_crop_mask, alpha=0.8, colors="blue")
    # show_drawn_masks_to_crop(drawn_masks)

    boxes = masks_to_boxes(to_crop_mask)  # (x1, y1, x2, y2)
    print(boxes)

    image_drawn_boxes = draw_bounding_boxes(segmented_image, boxes, colors="red")
    show_single_image(image_drawn_boxes)

    box_flatten = boxes.flatten().to(torch.int)

    # Crop the image using the bounding box coordinates
    x1, y1, x2, y2 = box_flatten[0].item(), box_flatten[1].item(), box_flatten[2].item(), box_flatten[3].item()
    cropped_image = segmented_image[:, y1:y2, x1:x2]
    print(cropped_image.size())
    show_single_image(cropped_image, axis=False)

    plt.imsave('segmented_leafs/corn_blight_cropped_image.jpg',
               cropped_image.numpy().transpose(1, 2, 0))
