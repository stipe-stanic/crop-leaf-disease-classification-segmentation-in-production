import numpy as np
import matplotlib.pyplot as plt
import cv2

from typing import List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from torchvision.utils import draw_bounding_boxes

from leaf_of_interest_algo import leaf_of_interest_selection
from segment_util import get_device, show_single_image, prepare_mask_to_crop, crop_segmented_image


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


def display_annotations(image: np.ndarray, anns: List[dict], axis: bool = True) -> None:
    """Display an image with annotated regions.

    :param image: The input image to be displayed
    :param anns: Masks which contain annotation information
    :param axis: Whether to show axis ticks and labels
    """

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
    image = cv2.imread("test_images/blight_corn_406.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    show_single_image(image, axis=False)

    img_height, img_width = image.shape[0], image.shape[1]

    sam_checkpoint = "../models_storage/pretrained/sam_vit_l_0b3195.pth"
    model_type = "vit_l"

    device = get_device()
    print(device)

    # Load segment anything model (SAM)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Define a grid of points for SAM mask generation
    point_grid = [np.array([
        [0.25, 0.25], [0.5, 0.25], [0.75, 0.25],  # top (left, center, right)
        [0.25, 0.5], [0.5, 0.5], [0.75, 0.5],  # middle (left, center, right)
        [0.25, 0.75], [0.5, 0.75], [0.75, 0.75],  # bottom (left, center, right)
        ])]

    # Display point grid on the image
    fig, ax = plt.subplots(figsize=(10, 10))
    show_point_grid(point_grid, image, ax)
    plt.show()

    min_mask_area = 0.05 * (img_height * img_width)  # to prevent small disconnected mask regions
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               points_per_side=None,
                                               point_grids=point_grid,
                                               stability_score_thresh=0.95,
                                               min_mask_region_area=min_mask_area
                                               )
    masks = mask_generator.generate(image)
    print(f'Number of masks: {len(masks)}')

    # Display annotation on the original image using generated masks
    display_annotations(image, masks)

    leaf_to_segment = leaf_of_interest_selection(masks, image)
    display_annotations(image, [leaf_to_segment])

    segmented_image = cv2.bitwise_and(image, image, mask=leaf_to_segment['segmentation'].astype(np.uint8))
    show_single_image(segmented_image, axis=False)

    # Split mask into a set of boolean masks.
    boxes, segmented_image = prepare_mask_to_crop(leaf_to_segment['segmentation'], segmented_image)

    # Draw bounding boxes on the segmented image
    image_drawn_boxes = draw_bounding_boxes(segmented_image, boxes, colors="red")
    show_single_image(image_drawn_boxes)

    cropped_image = crop_segmented_image(boxes, segmented_image)
    print(f'Cropped image size: {cropped_image.size()}')
    show_single_image(cropped_image, axis=False)
