import torch
import numpy as np
import torchvision.transforms.functional as F

from typing import Tuple
from torch import Tensor
from torchvision.ops import masks_to_boxes


def get_device() -> torch.device:
    """Get the device for running PyTorch computations.

    :returns: torch.device: The selected device (CPU or GPU).
    """

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    return device


def prepare_mask_to_crop(mask_to_crop: np.ndarray, segmented_image: np.ndarray) -> Tuple[Tensor, Tensor]:
    """Prepare the mask for cropping and convert segmented image to the appropriate format.

    :param mask_to_crop: Mask to be cropped
    :param segmented_image: Segmented image.
    :return: Tuple containing bounding boxes and segmented image tensor
    """

    mask_to_crop = np.squeeze(mask_to_crop)

    # Pixels can either     be in foreground or in background.
    obj_ids = np.unique(mask_to_crop)

    # Removing background pixels
    obj_ids = obj_ids[1:]

    # Splits the color-encoded mask into a set of boolean masks.
    to_crop_mask = mask_to_crop == obj_ids[:, None, None]

    segmented_image = np.transpose(segmented_image, (2, 0, 1))  # (channels x height x width)

    to_crop_mask = torch.from_numpy(to_crop_mask)
    segmented_image = torch.from_numpy(segmented_image)

    # Convert boolean masks to bounding boxes
    boxes = masks_to_boxes(to_crop_mask).to(torch.int)  # (x1, y1, x2, y2)
    print(f'Box coordinates: {boxes}')

    return boxes, segmented_image


def get_crop_coordinates(box_coords: Tensor, segmented_image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Calculate the crop coordinates for the segmented image based on bounding box coordinates.

    :param box_coords: Bounding box coordinates (x1, y1, x2, y2)
    :param segmented_image_shape: Shape of segmented image (channels, height, widht)
    :return: Crop coordinates with offsets
    """

    # Get the bounding box coordinates
    x1, y1, x2, y2 = box_coords[0].item(), box_coords[1].item(), box_coords[2].item(), box_coords[3].item()
    offset_y1, offset_y2, offset_x1, offset_x2 = -3, 3, -3, 3

    height, width = segmented_image_shape[1], segmented_image_shape[2]

    if y1 + offset_y1 > 0:
        y1 += offset_y1

    if y2 + offset_y2 < height:
        y2 += offset_y2

    if x1 + offset_x1 > 0:
        x1 += offset_x1

    if x2 + offset_x2 < width:
        x2 += offset_x2

    return x1, y1, x2, y2


def crop_segmented_image(boxes: Tensor, segmented_image: Tensor) -> Tensor:
    """Crop the segmented image based

    :param boxes: Bounding boxes (x1, y1, x2, y2)
    :param segmented_image: Segmented image tensor
    :return: Cropped segmented image tensor.
    """

    box_flatten = boxes.flatten().to(torch.int)

    x1, y1, x2, y2 = get_crop_coordinates(box_flatten, segmented_image.shape)
    cropped_image = segmented_image[:, y1:y2, x1:x2]

    return cropped_image
