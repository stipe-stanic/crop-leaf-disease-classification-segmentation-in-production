from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes=4):
        """Focal loss for multiclass classification.

        :param alpha: Weighting factor, positive class samples are given four times less weight than negative class samples.
        :param gamma: Focusing parameter, gamma > 1 increases emphasis on correctly classifying challenging scenarios.
        :param num_classes: Number of classes.
        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, cls_preds: Tensor, cls_targets: Tensor) -> Tensor:
        """Compute focal loss between cls_preds and cls_targets

        :param cls_preds: predicted class probabilities, sized [batch_size, num_classes]
        :param cls_targets: target class labels, sized [batch_size, num_classes]

        :returns: focal loss
        """

        # Converts target labels to one-hot encoded values
        t = F.one_hot(cls_targets, self.num_classes).float()  # [batch_size, num_classes]
        t = t.to(cls_preds.device)

        # Small value to avoid numerical instability
        epsilon = 1e-8

        p = torch.clamp(cls_preds.softmax(dim=1), min=epsilon, max=1 - epsilon)

        # Compute focal loss
        focal_loss = -self.alpha * (t * torch.log(p + epsilon) + (1 - t) * torch.log(1 - p + epsilon))
        focal_loss = focal_loss.sum()

        loss = focal_loss.mean()

        return loss
