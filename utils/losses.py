import torch
import torch.nn as nn


class WeightedCrossEntropy(object):
    def __init__(self, pos_wt, device):
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_wt)

    def compute_loss(self, pred, true):
        true = true.float().to(self.device)
        loss = self.loss_fn(pred.squeeze(), true)
        return loss
