# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

PPI (Protein-Protein Interaction) prediction using a graph neural network. The model classifies residue-level binding sites on proteins from structural and evolutionary features.

**Stack:** PyTorch + PyTorch Geometric + torch_sparse (SparseTensor) + scikit-learn + timm

## Key commands

```bash
# Full pipeline: train + evaluate
bash run.sh

# Train with 5-fold cross-validation (includes temperature scaling after each fold)
python demo.py

# Fit temperature scaling on a calibration set (standalone CLI)
python Fit_T.py --model_dir ../data/Saved_model --data_path ../data/DSet_72 --save_path ../data/Saved_model

# Evaluate k-fold calibrated models on a test set
python Val.py
```

There is no `requirements.txt` or setup script. Dependencies must be installed manually.

## Architecture

### Data pipeline (`Data.py`)

Datasets are stored as **8 ordered pickle files** per directory — one file per feature type. The fixed order is: `aaindex`, `BLOSUM`, `dssp`, `edge`, `ESMC`, `label`, `pse_1a`, `res_atom_1a`. `PPIData.load_data()` loads these by keyword matching and assembles per-protein dicts.

- **`PPIDataset`** wraps a list of protein dicts; each sample is a single protein graph.
- **`sparse_collate`** merges multiple protein graphs into one large disconnected graph (offsetting edge indices by cumulative node count) so a batch is a single `SparseTensor` adjacency.
- Input feature dimensions per residue: AAINDEX (566), ESM-C (1152), DSSP (14), BLOSUM62 (20), pseudo-position (1), residue atoms (7).

### Model (`model/model.py`)

`PPI` is a graph transformer for node-level binary classification:

1. **Feature encoding:** AAINDEX goes through a residual encoder (`aa_encoder`); DSSP + BLOSUM + pseudo-position + residue-atom are concatenated. Both streams are fused with ESM-C via a **gated fusion** (`Gated_Fuse`) and projected to `hid_dim` (128).
2. **Graph convolutions:** A stack of `PPIBlock` layers (default 4). Each block = `GPSConv` wrapping `TransformerConv` (8 heads, edge_dim=1) + DropPath + LayerNorm + residual.
3. **Classifier:** 2-layer MLP (hid_dim → hid_dim/2 → 1) with GELU and dropout.

### Optimizer (`model/Optimizer.py`)

**SophiaG** — a second-order optimizer that uses diagonal Hessian estimates. Unlike standard optimizers, it requires calling `optimizer.update_hessian()` periodically (every 10 steps in this code) and passing `bs=BATCH_SIZE` to `optimizer.step()`.

### Loss (`utils/losses.py`)

**`HybridLoss`** combines weighted BCE and Tversky loss. Focal loss is available but default weight=0. The config controls component weights (`BCE_WEIGHT`, `Tversky_WEIGHT`) and the Tversky α/β (FP/FN penalty).

### Training (`demo.py`)

`cross_validate()` runs 5-fold CV:

- Splits data 80/10/10 train/test/calib by protein
- Each fold: warmup LR schedule (5 epochs) → ReduceLROnPlateau on AUPRC → early stopping (patience=8)
- After each fold converges, applies **temperature scaling** on the calibration set and saves both `Model_fold{N}.pth` (uncalibrated) and `Model_fold{N}_calibrated.pth`

### Calibration (`Fit_T.py`)

Fits a single temperature parameter `T` via L-BFGS on BCE loss, then saves the calibrated model with `T` and the optimal classification threshold `r`. Usable as a CLI with `--model_dir`, `--data_path`, `--save_path`.

### Evaluation (`Val.py`)

`test_kfold_models()` loads calibrated fold models, runs inference, and reports per-fold and mean metrics. `validate()` draws ROC/PR curves if `draw_plots=True`. The script in `__main__` runs evaluation on `config.VAL3` by default.

### Config (`config.py`)

All hyperparameters and paths in one file. Key settings: `NUM_LAYER=4`, `HEADS=8`, `gcn_hid_dim=128`, `EPOCHS=100`, `BATCH_SIZE=32`, `LEARNING_RATE=4e-4`, `PATIENCE=8`, `K_FOLDS=5`, `F_BETA=0.7`. Loss weights: `A=0.3` (FP), `B=0.97` (FN), `BCE_WEIGHT=0.3`, `Tversky_WEIGHT=0.7`. Data paths are relative to the project root (`../data/...`).

## Common patterns

- **Batch handling:** Every batch is a dict with keys `AA`, `esm_c`, `dssp`, `BLOSUM`, `pse`, `res_atom`, `adj`, `y`. Models and eval code move tensors to `config.DEVICE` at call time.
- **Model loading:** Calibrated models are saved as `{'model': state_dict, 'T': tensor, 'r': tensor}`. Load with `_load_model()` in `Val.py`.
- **Threshold optimization:** `find_best_threshold_by_f_beta()` finds the decision threshold maximizing F-beta (default β=0.7, biasing toward recall).
- **Metrics dict:** Standard return format with accuracy, MCC, ROC-AUC, PR-AUC, F1, precision, recall, specificity, threshold, and confusion matrix.

## Known issues

- **`TverskyLoss.forward()`** (`utils/losses.py:44-54`) contains a nested duplicate `forward` definition that is dead code — the outer body (lines 55-60) is what actually executes. The inner function has no effect but is harmless.
