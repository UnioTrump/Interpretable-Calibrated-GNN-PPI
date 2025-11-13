import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropy(nn.Module):
    def __init__(self, pos_wt):
        super(WeightedCrossEntropy, self).__init__()
        self.pos_weight = pos_wt

    def forward(self, pred, true):

        loss = F.binary_cross_entropy_with_logits(pred.squeeze(), true, pos_weight=self.pos_weight)
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, true):
        pred = torch.sigmoid(pred)    # after sigmoid

        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - pred), self.gamma) * torch.log(pred + eps) * true
        loss_0 = -1 * (1 - self.alpha) * torch.pow(pred, self.gamma) * torch.log(1 - pred + eps) * (1 - true)
        loss = loss_0 + loss_1
        return torch.mean(loss)

class TverskyLoss(nn.Module):
    def __init__(self, alpha, beta, smooth=1e-6):
        super(TverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, true):

        pred = torch.sigmoid(pred).squeeze()
        true = true.view(-1)
        pred = pred.view(-1)

        TP = (pred * true).sum()
        FP = ((1 - true) * pred).sum()
        FN = (true * (1 - pred)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky_index).pow(0.75)


class HybridLoss(nn.Module):
    def __init__(self, alpha, beta, pos_wt, bce_weight, focal_weight, tversky_weight, smooth=1e-5):
        super(HybridLoss, self).__init__()

        self.bce = WeightedCrossEntropy(pos_wt=pos_wt)
        self.focal = FocalLoss(alpha=0.25, gamma=2)
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, pred, true):
        pred = pred.float()
        true = true.float()
        bce_loss = self.bce(pred, true)
        focal_loss = self.focal(pred, true)
        tversky_loss = self.tversky(pred, true)
        return self.focal_weight * focal_loss + self.tversky_weight * tversky_loss + self.bce_weight * bce_loss

if __name__ == '__main__':
    pred = torch.tensor([-3., 4., -7., 9., 2.])
    true = torch.tensor([0., 1., 0., 1., 0.])

    loss_fun = HybridLoss(alpha=0.7, beta=0.7, bce_weight=1, focal_weight=0, tversky_weight=0.5)
    loss=loss_fun(pred, true)
    print(loss)
