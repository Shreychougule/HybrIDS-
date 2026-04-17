# Hybrid NIDS (HybrIDS)

Hybrid NIDS is a network intrusion detection project that combines a ProtoNet-style embedding model with an XGBoost classifier for robust multi-class attack detection and better handling of low-confidence/unknown samples.

The repository includes:
- `HNIDS.ipynb`: main notebook for experimentation/training/evaluation flow.
- `prediction.py`: inference module for batch prediction from preprocessed features.
- `models/`: saved artifacts (centroids, thresholds, hybrid config, and XGBoost model).
- `demo_report_20251104_202007.pdf`: demo/report document.

## Project Overview

The inference pipeline uses:
- **Proto distance branch**: computes fused distance (Euclidean + cosine) from learned embeddings to class centroids.
- **XGBoost branch**: predicts class probabilities from Proto embeddings.
- **Hybrid fusion**: combines both branches using configured weights (`w_proto`, `w_xgb`).
- **Gating + tiering**: marks samples as `known`, `rare`, or `unknown` using distance and confidence thresholds.

Key runtime settings are stored in `models/hybrid_config.json`, including:
- `tau_dist` for distance-based acceptance.
- `p_thresh` for confidence-based acceptance.
- fusion weights and normalization mode.

## Quick Guide

### 1) Prerequisites

- Python 3.9+ recommended
- Install required libraries:

```bash
pip install numpy torch xgboost
```

If you use Jupyter for `HNIDS.ipynb`:

```bash
pip install notebook
```

### 2) Clone and move into project

```bash
git clone https://github.com/Shreychougule/HybrIDS-.git
cd "HybrIDS-"
```

### 3) Ensure model artifacts are present

The following files should exist in `models/`:
- `euc_final_centroids.pt`
- `rejection_thresholds.pt`
- `hybrid_config.json`
- `xgb_on_proto_emb.model`

### 4) Run notebook workflow

Open and run:

```bash
jupyter notebook HNIDS.ipynb
```

Use the notebook to train/evaluate and prepare the same preprocessing pipeline used for inference.

### 5) Use prediction module

`prediction.py` expects:
- a loaded/initialized ProtoNet object named `model` in memory,
- input features already preprocessed exactly as during training.

Example usage pattern:

```python
import numpy as np
from prediction import predict_batch_from_features

# X_test should be preprocessed features with shape (N, F)
X_test = np.random.rand(8, 20).astype(np.float32)
out = predict_batch_from_features(X_test, norm='per_sample')

print(out["pred"])
print(out["tier"])
```

## Notes

- Keep paths consistent with the artifact names expected by `prediction.py`.
- If your training used global normalization stats, provide `global_norm_stats.pt`.
- For production use, package preprocessing + model loading together to avoid feature mismatch.