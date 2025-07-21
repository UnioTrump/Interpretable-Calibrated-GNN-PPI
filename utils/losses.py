import torch
import torch.nn.functional as F


class WeightedCrossEntropy(object):
    def __init__(self, pos_wt, device):
        self.device = device
        self.pos_weight = pos_wt.to(self.device) if torch.is_tensor(pos_wt) else torch.tensor(pos_wt, device=self.device)

    def compute_loss(self, pred, true):
        true = true.float().to(self.device)
        
        loss = F.binary_cross_entropy_with_logits(
            pred.squeeze(), 
            true, 
            pos_weight=self.pos_weight
        )
        return loss
