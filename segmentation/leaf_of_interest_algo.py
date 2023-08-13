import numpy as np

from typing import List


def distance_to_image_center(bbox: np.ndarray, image_center_x: float, image_center_y: float) -> float:
    """Calculate the Euclidean distance between the center of a bounding box and the center of the image.

    :param bbox: Bounding box coordinates in [x_min, y_min, x_max, y_max] format
    :param image_center_x: X-coordinate of the image center
    :param image_center_y: Y-coordinate of the image center
    :return: Euclidean distance
    """

    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2

    return np.sqrt((x_center - image_center_x)**2 + (y_center - image_center_y)**2)


def leaf_of_interest_selection(masks: List[dict], image: np.ndarray) -> dict:
    """Select leaf to segment from a list of masks based on criteria.

    :param masks: List of masks, each mask contains segmentation information
    :param image: Input image
    :return: Mask representing selected leaf
    """

    if len(masks) == 0:
        raise ValueError("No masks found")

    image_center_x = image.shape[1] / 2
    image_center_y = image.shape[0] / 2

    for i, mask in enumerate(masks):
        mask['id'] = i + 1
        mask['criterion_winner'] = []

    # Point ratios for each criterion
    p_height = 0.2
    p_width = 0.2
    p_area = 0.3
    p_center_distance = 0.15
    p_accuracy = 0.15

    # Find the mask with the biggest continuous masked area
    mask_area = max(masks, key=lambda x: x['area'])
    mask_area['criterion_winner'].append('area')

    # Find the mask with the maximum width, height and accuracy
    mask_width = max(masks, key=lambda x: x['bbox'][2])  # bbox is in XYWH format
    mask_width['criterion_winner'].append('width')

    mask_height = max(masks, key=lambda x: x['bbox'][3])
    mask_height['criterion_winner'].append('height')

    mask_accuracy = max(masks, key=lambda x: x['stability_score'])
    mask_accuracy['criterion_winner'].append('accuracy')

    # Find the mask closest to the center of the image
    mask_center = min(masks, key=lambda x: distance_to_image_center(x['bbox'], image_center_x, image_center_y))
    mask_center['criterion_winner'].append('center')

    mask_criterion_winners = [mask_area, mask_width, mask_height, mask_accuracy, mask_center]

    # Group masks by ID to filter duplicates
    mask_grouped = {}
    for mask in mask_criterion_winners:
        if mask['id'] not in mask_grouped.keys():
            mask_grouped[mask['id']] = mask

    # Find the mask with the highest combined score
    mask_best = None
    highest_score = 0
    for key in mask_grouped.keys():
        total_score = 0
        mask = mask_grouped[key]

        for crit in mask['criterion_winner']:
            if crit == 'area':
                total_score += 100 * p_area
            elif crit == 'center':
                total_score += 100 * p_center_distance
            elif crit == 'width':
                total_score += 100 * p_width
            elif crit == 'height':
                total_score += 100 * p_height
            elif crit == 'accuracy':
                total_score += 100 * p_accuracy

        if total_score > highest_score:
            highest_score = total_score
            mask_best = mask

    return mask_best
