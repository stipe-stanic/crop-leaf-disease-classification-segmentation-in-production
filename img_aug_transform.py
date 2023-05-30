from typing import Tuple

import numpy as np
from imgaug import augmenters as iaa
import cv2
from PIL import Image


class CustomCLAHE(object):
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        """Initializes an object of CustomCLAHE class with specified parameters.

        :param clip_limit: The contrast limit.
        :param tile_grid_size: The size of the grid.
        """

        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image) -> Image:
        """Applies Contrast Limited Adaptive Histogram Equalization to the input image.

        :param img: The input image.
        :returns: The equalized image.
        """

        img = np.array(img)

        # Converts the image from BGR to LAB color space
        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Applies CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_l_channel = clahe.apply(l_channel)

        # Merges the equalized L channel with original a and b channels
        equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])

        # Converts the equalized LAB image back to BGR color space
        equalized_bgr_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

        return Image.fromarray(equalized_bgr_image)


class ImgAugGenerate:
    def __init__(self):
        """Initializes an object of ImgAugTransform class with a sequence of image augmentation operations"""

        self.aug = iaa.Sequential([
            iaa.Resize({'height': 224, 'width': 'keep-aspect-ratio'}),
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
        ])

    def __call__(self, img: Image) -> np.ndarray:
        """Applies the sequence of image augmentation operations to the input image

        :param img: The input image.
        :returns: The augmented image.
        """

        img = np.array(img)

        # Applies the image augmentation operations to the input image and returns the augmented image
        return self.aug.augment_image(img)
