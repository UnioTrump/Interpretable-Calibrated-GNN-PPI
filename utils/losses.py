import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


import matplotlib.pyplot as plt

class WeightedCrossEntropy(nn.Module):
    def __init__(self, pos_wt, device):
        super(WeightedCrossEntropy, self).__init__()
        self.pos_weight = pos_wt.to(self.device) if torch.is_tensor(pos_wt) else torch.tensor(pos_wt)

    def compute_loss(self, pred, true):
        # true=torch.LongTensor(true)
        true = true.float().to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(), 
            true, 
            pos_weight=self.pos_weight
        )
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)    # after sigmoid

        eps = 1e-7
        loss_1 = -1 * self.alpha * torch.pow((1 - preds), self.gamma) * torch.log(preds + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(preds, self.gamma) * torch.log(1 - preds + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)

class TverskyLoss(nn.Module):
    def __init__(self, alpha, beta, device, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, true):

        true = true.float().to(self.device)
        pred = torch.sigmoid(pred).squeeze()
        true = true.view(-1)
        pred = pred.view(-1)

        TP = (pred * true).sum()
        FP = ((1 - true) * pred).sum()
        FN = (true * (1 - pred)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky_index).pow(0.75)


class HybridLoss(nn.Module):
    def __init__(self, alpha, beta, device, focal_weight, tversky_weight, smooth=1e-5):
        super(HybridLoss, self).__init__()

        self.focal = FocalLoss(alpha=0.25, gamma=2)
        self.tversky = TverskyLoss(alpha, beta, device, smooth)
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, pred, true):
        focal_loss = self.focal(pred, true)
        tversky_loss = self.tversky(pred, true)
        return self.focal_weight * focal_loss + self.tversky_weight * tversky_loss

if __name__ == '__main__':
    pred = torch.tensor([-3., 4., -7., 9., 2.])
    labels = torch.tensor([0., 1., 0., 1., 0.])
    # 固定beta值，观察alpha变化时的损失
    fixed_beta = 0.5
    alphas = np.linspace(0.1, 0.9, 50)  # 从0.1到0.9均匀分布的50个点
    loss_values_alpha = []

    for alpha in alphas:
        tversky_loss = TverskyLoss(alpha=alpha, beta=fixed_beta, device='cpu')
        loss = tversky_loss(pred, labels).item()
        loss_values_alpha.append(loss)

    # 绘制alpha变化时的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, loss_values_alpha, label=f'Fixed Beta: {fixed_beta}')
    plt.xlabel('Alpha')
    plt.ylabel('Tversky Loss')
    plt.title('Tversky Loss for Different Alpha Values (Beta Fixed)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 固定alpha值，观察beta变化时的损失
    fixed_alpha = 0.5
    betas = np.linspace(0.1, 0.9, 50)  # 从0.1到0.9均匀分布的50个点
    loss_values_beta = []

    for beta in betas:
        tversky_loss = TverskyLoss(alpha=fixed_alpha, beta=beta, device='cpu')
        loss = tversky_loss(pred, labels).item()
        loss_values_beta.append(loss)

    # 绘制beta变化时的损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(betas, loss_values_beta, label=f'Fixed Alpha: {fixed_alpha}')
    plt.xlabel('Beta')
    plt.ylabel('Tversky Loss')
    plt.title('Tversky Loss for Different Beta Values (Alpha Fixed)')
    plt.legend()
    plt.grid(True)
    plt.show()

