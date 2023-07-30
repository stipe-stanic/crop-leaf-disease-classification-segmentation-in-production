import cv2
import numpy as np

from typing import Tuple
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
        lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Applies CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        equalized_l_channel = clahe.apply(l_channel)

        # Merges the equalized L channel with original a and b channels
        equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])

        # Converts the equalized LAB image back to BGR color space
        equalized_bgr_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2RGB)

        return Image.fromarray(equalized_bgr_image)