import numpy as np
from imgaug import augmenters as iaa
from torchvision.transforms import functional as F
import cv2
from PIL import Image

class CustomCLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = img.astype(np.uint8)
        else:
            img = np.array(img)

        lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_l_channel = clahe.apply(l_channel)

        equalized_lab_image = cv2.merge([equalized_l_channel, a_channel, b_channel])
        equalized_bgr_image = cv2.cvtColor(equalized_lab_image, cv2.COLOR_LAB2BGR)

        return Image.fromarray(equalized_bgr_image)


class ImgAugTransform:
    def __init__(self):
        """Initializes an object of ImgAugTransform class with a sequence of image augmentation operations"""

        # Boilerplate transformations
        self.aug = iaa.Sequential([
            iaa.Resize({'height': 224, 'width': 'keep-aspect-ratio'}),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.1)),
                              iaa.CoarseDropout(0.1, size_percent=0.5)
                          ])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.Sometimes(0.25, iaa.Grayscale())
        ])

    def __call__(self, img):
        """Applies the sequence of image augmentation operations to the input image

        :param img: the input image
        :returns: the augmented image
        """

        img = np.array(img)
        return self.aug.augment_image(img)
