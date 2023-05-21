from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

from torch.autograd import Variable


def one_hot_embedding(labels: LongTensor, num_classes: int) -> Tensor:
    """Embedding labels to one-hot form.

    :param labels: Class labels, sized [N,].
    :param num_classes: Number of classes.

    :returns: Encoded labels, sized [N, #classes].
    """

    y = torch.eye(num_classes)  # [D, D]
    return y[labels]


class FocalLoss(nn.Module):
    def __init__(self, num_classes=4):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Focal loss.

        :param x: tensor sized [N, D]
        :param y: tensor sized [N, ]

        :returns: focal loss
        """

        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)  # [N, 21]
        t = t[:, 1:]  # exclude background
        t = Variable(t).cuda()  # [N, 20]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)         # pt = p if t > 0 else 1-p
        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x: Tensor, y: Tensor) -> Tensor:
        """Focal loss alternative.

        :param x: tensor sized [N, D]
        :param y: tensor sized [N, ]

        :returns: focal loss
        """

        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), 1 + self.num_classes)
        t = t[:, 1:]
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds: Tensor, loc_targets: Tensor, cls_preds: Tensor, cls_targets: Tensor) -> Tensor:
        """Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        :param loc_preds: predicted locations, sized [batch_size, #anchors, 4]
        :param loc_targets: encoded target locations, sized [batch_size, #anchors, 4]
        :param cls_preds: predicted class confidences, sized [batch_size, #anchors, #classes]
        :param cls_targets: encoded target labels, sized [batch_size, #anchors]

        :returns: loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets)
        """

        # batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N, #anchors]
        num_pos = pos.data.long().sum()

        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N, #anchors, 4]
        masked_loc_preds = loc_preds[mask].view(-1, 4)      # [#pos, 4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos, 4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        # cls_loss = FocalLoss(loc_preds, loc_targets)
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0]/num_pos, cls_loss.data[0]/num_pos), end=' | ')
        loss = (loc_loss+cls_loss) / num_pos
        return loss
