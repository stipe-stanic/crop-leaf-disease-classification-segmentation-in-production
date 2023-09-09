import torch
import cv2
import numpy as np

from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from app.segmentation.segment_util import prepare_mask_to_crop, crop_segmented_image
from app.segmentation.leaf_ROI_algo import leaf_of_interest_selection


def segment_object(input: Image, model_checkpoint: str) -> Image:
    """Segment a leaf object in an image using a provided SAM checkpoint.

    :param input: Input image
    :param model_checkpoint: Path to the model checkpoint for segmentation
    :return: Segmented leaf
    """

    sam_checkpoint = model_checkpoint
    model_type = sam_checkpoint.split('/')[-1][4:9]  # example: 'vit_l'

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    image = np.array(input)
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
    if len(masks) == 0:
        print("No masks found")

    leaf_to_segment = leaf_of_interest_selection(masks, image)
    segmented_image = cv2.bitwise_and(image, image, mask=leaf_to_segment['segmentation'].astype(np.uint8))

    boxes, segmented_image = prepare_mask_to_crop(leaf_to_segment['segmentation'], segmented_image.copy())
    if boxes.shape[0] == 0:
        print("No bounding box found")

    cropped_image = crop_segmented_image(boxes, segmented_image)
    pil_cropped_image = Image.fromarray(cropped_image.numpy().transpose(1, 2, 0))

    return pil_cropped_image
