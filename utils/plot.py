import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_loss_curves(train_losses, val_losses, save_path='loss_curves.png'):
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if val_losses:
        best_val_epoch = np.argmin(val_losses)
        best_val_loss = val_losses[best_val_epoch]
        plt.axvline(x=best_val_epoch + 1, color='gray', linestyle='--')
        plt.scatter(best_val_epoch + 1, best_val_loss, marker='*', s=150, color='gold', zorder=5, label=f'Best Val Loss: {best_val_loss:.4f} at Epoch {best_val_epoch +1}')

    plt.legend()
    plt.savefig(save_path, dpi=300)
    print(f"Loss curve saved to: {save_path}")
    plt.close()

def save_metrics_to_txt(metrics, Dset_name):
    file = f"{Dset_name}_metrics.txt"
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Metrics for {Dset_name}\n")
        f.write("=" * 50 + "\n")
        
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                f.write(f"{key.replace('_', ' ').title()}: {val:.4f}\n")
            elif isinstance(val, dict):
                f.write(f"{key.replace('_', ' ').title()}:\n")
                for sub_key, sub_val in val.items():
                    f.write(f"  {sub_key.upper()}: {sub_val}\n")
            else:
                f.write(f"{key.replace('_', ' ').title()}: {val}\n")
        
        f.write("=" * 50 + "\n")
        f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Metrics saved to: {file}")

def draw(y_true, y_pred, save_dir='./plots'):
    os.makedirs(save_dir, exist_ok=True)
    # ========== ROC 曲线 ==========
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # ========== PR 曲线 ==========
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPRC = {auprc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Plots saved to {save_dir}/")