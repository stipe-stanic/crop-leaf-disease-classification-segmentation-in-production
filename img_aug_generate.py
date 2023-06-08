import numpy as np

from imgaug import augmenters as iaa
from PIL import Image


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
