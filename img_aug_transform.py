import numpy as np
from imgaug import augmenters as iaa

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