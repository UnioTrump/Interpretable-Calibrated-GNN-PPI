import torch
import torch.nn.functional as F


class WeightedCrossEntropy(object):
    def __init__(self, pos_wt, device):
        self.device = device
        # 将 pos_wt 存储为类的属性，而不是创建 BCEWithLogitsLoss 实例
        # 确保它是一个可以在需要时使用的标量或张量
        self.pos_weight = pos_wt.to(self.device) if torch.is_tensor(pos_wt) else torch.tensor(pos_wt, device=self.device)

    def compute_loss(self, pred, true):
        # 确保 true 是 float 类型并位于正确的设备上
        true = true.float().to(self.device)
        
        # 直接调用函数版本，在每次计算时都明确传入 pos_weight
        # 这是更健壮、更清晰的做法，可以避免在__init__中因尺寸不匹配引发的潜在问题
        loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(), 
            true, 
            pos_weight=self.pos_weight
        )
        return loss
