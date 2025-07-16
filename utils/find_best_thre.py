import numpy as np
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef

def find_best_threshold_by_f_beta(y_true, y_scores, num_threshold, beta=1.5):
    """
        通过Fβ寻找最佳阈值

        Args:
            beta: precision和recall的权重值
    """

    thresholds = np.linspace(0, 1, num_threshold)
    y_true = y_true.cpu().numpy()
    y_scores = y_scores.cpu().numpy()
    f_beta_values = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        precisions = precision_score(y_true, y_pred, zero_division=0)
        recalls = recall_score(y_true, y_pred)
        f_beta = (1 + beta**2)*(precisions * recalls)/(beta**2*precisions+recalls) if (precisions + recalls) != 0 else 0
        f_beta_values.append(f_beta)

    best_idx = np.argmax(f_beta_values)
    best_threshold = thresholds[best_idx]
    best_f_beta = f_beta_values[best_idx]

    return best_threshold, best_f_beta