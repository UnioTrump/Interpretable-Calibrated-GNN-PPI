from torch import nn
import torch
from utils import find_best_threshold_by_f_beta
import config
from tqdm import tqdm

device = config.DEVICE

@torch.no_grad()
def getlogits(model, val_loader):
    model.eval()
    logits, targets = [], []
    for batch in tqdm(val_loader, total=len(val_loader)):
        batch = {
            k: v.to(config.DEVICE) if (torch.is_tensor(v) or hasattr(v, 'to')) else v
            for k, v in batch.items()
        }
        out = model(ax=batch['AA'], bx=batch['esm_c'], cx=batch['dssp'], dx=batch['BLOSUM'], ex=batch['pse'],
                    fx=batch['res_atom'], adj=batch['adj'])

        logits.append(out)
        targets.append(batch['y'].float())

    logits_t = torch.cat(logits, dim=0)
    targets_t = torch.cat(targets, dim=0)

    return logits_t, targets_t


@torch.no_grad()
def test_T(logits, targets, T):
    """
        Search the best threshold on validation set after temperature scaling.
    """

    cal_logits = logits / T
    probs = torch.sigmoid(cal_logits)

    threshold = find_best_threshold_by_f_beta(targets, probs, num_threshold=100)

    return threshold

class Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))

    @property
    def T(self):
        return torch.exp(self.log_T)

    def forward(self, logits):
        return logits / self.T

def fit_T(logits, labels):
    scaler = Temp().to(device)
    optimizer_T = torch.optim.LBFGS([scaler.T], lr=config.LEARNING_RATE, max_iter=50)
    def closure():
        optimizer_T.zero_grad()
        scaled_logits = scaler(logits)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.1))(scaled_logits, labels.float())
        loss.backward()
        return loss.detach()
    optimizer_T.step(closure)
    return scaler.T.detach().clone()
