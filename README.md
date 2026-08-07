# An Interpretable and Calibrated Attention-Based Graph Neural Network for Protein–Protein Interaction Site Identification

Protein–Protein Interaction (PPI) site prediction at residue level using a Graph Neural Network with multi-modal protein features.

## Overview

![a](./Fig/Fig1.tif)

This project predicts PPI binding sites via a **GraphGPS** architecture with Transformer convolutions, integrating six complementary feature channels per residue:

| Feature | Dim | Description |
|---------|-----|-------------|
| AAINDEX | 566 | Physicochemical property indices |
| ESM-C | 1152 | Pre-trained protein language model embeddings |
| DSSP | 14 | Secondary structure profiles |
| BLOSUM62 | 20 | Evolutionary substitution matrix |
| Pseudo Position | 1 | Sequence position encoding |
| Residue Atom | 7 | Atomic composition features |

## Project Structure

```
.
├── config.py           # Hyperparameters & paths
├── Data.py             # Dataset loading, batching, train/val/test split
├── demo.py             # K-fold training + temperature calibration
├── Fit_T.py            # Temperature scaling & threshold search
├── Val.py              # Evaluation on external test sets
├── run.sh              # Example run script
├── model/
│   ├── model.py        # PPI GNN (GPSConv + gated fusion)
│   └── Optimizer.py    # SophiaG optimizer
├── utils/
│   ├── losses.py       # HybridLoss (Tversky + BCE), FocalLoss
│   ├── metrics.py      # AUPRC, AUROC, F1, MCC, threshold search
│   └── plot.py         # Loss curves & ROC/PR plots
└── RAW DATA/           # Raw data directory
```

## Setup

```bash
conda env create -f environment.yml
conda activate PPI
```

## Usage

**Train with 5-fold cross-validation on Train1958:**

```bash
python demo.py
```

**Evaluate on external test sets:**

```bash
python Val.py --data_path /path/to/test_data
```

**Run the full pipeline:**

```bash
bash run.sh
```

## Cite Our Paper
If you consider our work to be of value and wish to cite it, please cite using the reference below: