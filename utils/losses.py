import torch
import torch.nn.functional as F


class WeightedCrossEntropy(object):
    def __init__(self, pos_wt, device):
        self.device = device
        self.pos_weight = pos_wt.to(self.device) if torch.is_tensor(pos_wt) else torch.tensor(pos_wt, device=self.device)

    def compute_loss(self, pred, true):
        # true=torch.LongTensor(true)
        true = true.float().to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(), 
            true, 
            pos_weight=self.pos_weight
        )
        return loss


class TverskyLoss(object):
    def __init__(self, alpha, beta, device, smooth=1e-5):
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def compute_loss(self, pred, true):
        # true = torch.LongTensor(true)
        true = true.float().to(self.device)
        pred = torch.sigmoid(pred).squeeze()

        # Flatten label and prediction tensors
        true = true.view(-1)
        pred = pred.view(-1)

        TP = (pred * true).sum()
        FP = ((1 - true) * pred).sum()
        FN = (true * (1 - pred)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - tversky_index
        return loss


class HybridLoss(object):
    def __init__(self, pos_wt, alpha, beta, device, ce_weight=0.5, tversky_weight=0.5, smooth=1e-5):
        self.weighted_ce = WeightedCrossEntropy(pos_wt, device)
        self.tversky_loss = TverskyLoss(alpha, beta, device, smooth)
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight

    def compute_loss(self, pred, true):
        ce_loss = self.weighted_ce.compute_loss(pred, true)
        tversky_loss = self.tversky_loss.compute_loss(pred, true)
        return self.ce_weight * ce_loss + self.tversky_weight * tversky_loss
