# ------------------ Prediction Module (paste into notebook / service) ------------------
import os, json
import numpy as np
import torch
import torch.nn.functional as F

# ---------- CONFIG (filenames must match your project artifacts) ----------
CENTROIDS_PATH     = "euc_final_centroids.pt"
REJ_THRESH_PATH    = "rejection_thresholds.pt"
XGB_MODEL_PATH     = "xgb_on_proto_emb.model"
HYBRID_CONFIG_PATH = "hybrid_config.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Helper: safe load thresholds/centroids that may contain numpy scalars ----------
def safe_torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        import torch.serialization, numpy as _npy
        torch.serialization.add_safe_globals([_npy.core.multiarray.scalar])
        return torch.load(path, map_location=map_location)

# ---------- Try to reuse compute_fused_distances if present; otherwise define local equivalent ----------
if 'compute_fused_distances' in globals():
    _compute_fused = globals()['compute_fused_distances']
else:
    # local implementation consistent with Cells 5/6/11
    def _compute_fused(embeddings, centroids, model, norm='per_sample', global_stats=None):
        """
        embeddings: torch.Tensor (N, z) on any device or CPU
        centroids: torch.Tensor (C, z)
        returns: d_fused (cpu tensor), d_e (cpu tensor), d_c (cpu tensor)
        """
        device_local = embeddings.device if isinstance(embeddings, torch.Tensor) else device
        emb = embeddings.to(device_local)
        cent = centroids.to(device_local)

        # Euclidean and Cosine
        d_e = torch.cdist(emb, cent)  # (N, C) Euclidean
        d_e = d_e * d_e               # squared Euclidean (as used in your merged pipeline)
        # cosine distance
        emb_norm = emb / (emb.norm(dim=1, keepdim=True) + 1e-8)
        cent_norm = cent / (cent.norm(dim=1, keepdim=True) + 1e-8)
        d_c = 1.0 - torch.matmul(emb_norm, cent_norm.t())

        if norm == 'per_sample':
            # per-sample min-max
            d_e_min = d_e.min(dim=1, keepdim=True)[0]
            d_e_max = d_e.max(dim=1, keepdim=True)[0]
            d_e_norm = (d_e - d_e_min) / (d_e_max - d_e_min + 1e-8)

            d_c_min = d_c.min(dim=1, keepdim=True)[0]
            d_c_max = d_c.max(dim=1, keepdim=True)[0]
            d_c_norm = (d_c - d_c_min) / (d_c_max - d_c_min + 1e-8)
        else:
            assert global_stats is not None, "global_stats required for global normalization"
            min_e = global_stats['min_e'].to(device_local)
            max_e = global_stats['max_e'].to(device_local)
            min_c = global_stats['min_c'].to(device_local)
            max_c = global_stats['max_c'].to(device_local)
            d_e_norm = (d_e - min_e) / (max_e - min_e + 1e-8)
            d_c_norm = (d_c - min_c) / (max_c - min_c + 1e-8)

        # alpha
        if hasattr(model, 'alpha_param'):
            alpha = torch.sigmoid(model.alpha_param).to(device_local)
        else:
            alpha = torch.tensor(0.5, device=device_local)

        d_fused = alpha * d_e_norm + (1.0 - alpha) * d_c_norm
        # move to CPU for downstream (matching your project's pattern)
        return d_fused.detach().cpu(), d_e.detach().cpu(), d_c.detach().cpu()

# ---------- Load artifacts (centroids, thresholds, hybrid config) ----------
if not ('model' in globals()):
    raise RuntimeError("ProtoNet `model` object not found in the notebook. Instantiate/load ProtoNet before using this predict module.")

model.to(device).eval()

if not os.path.exists(CENTROIDS_PATH):
    raise FileNotFoundError(f"Centroids artifact not found: {CENTROIDS_PATH}")
centroids = safe_torch_load(CENTROIDS_PATH, map_location=device)
if not isinstance(centroids, torch.Tensor):
    centroids = torch.tensor(np.asarray(centroids), dtype=torch.float32)
centroids = centroids.to(device)

import torch.serialization, numpy as np

# --- Safe, compatible loader for mixed PyTorch/Numpy pickles ---
try:
    torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar])
    loaded = torch.load(REJ_THRESH_PATH, map_location='cpu', weights_only=False)
except Exception as e:
    print("⚠️ Safe load failed, retrying with relaxed settings due to:", e)
    torch.serialization.add_safe_globals([np.dtype, np.core.multiarray.scalar])
    loaded = torch.load(REJ_THRESH_PATH, map_location='cpu', weights_only=False)


best_w = None
if os.path.exists(HYBRID_CONFIG_PATH):
    with open(HYBRID_CONFIG_PATH, 'r') as fh:
        cfg = json.load(fh)
        best_w = float(cfg.get('w_proto', cfg.get('w_proto', 0.5)))
        tau_dist = float(cfg.get('tau_dist', tau_dist))
        p_thresh = float(cfg.get('p_thresh', p_thresh))
if best_w is None:
    best_w = 0.5  # fallback

# load xgboost model lazily (if present)
_have_xgb = False
_booster = None
try:
    import xgboost as xgb
    if os.path.exists(XGB_MODEL_PATH):
        _booster = xgb.Booster()
        _booster.load_model(XGB_MODEL_PATH)
        _have_xgb = True
except Exception:
    _have_xgb = False

# ---------- Optional: load global_stats if you used global normalization (else None) ----------
global_stats = None
# if you saved global_stats as 'global_norm_stats.pt' in your pipeline, load it here:
if os.path.exists('global_norm_stats.pt'):
    stats = safe_torch_load('global_norm_stats.pt', map_location='cpu')
    # ensure torch tensors
    global_stats = {
        'min_e': torch.tensor(stats['min_e'], dtype=torch.float32),
        'max_e': torch.tensor(stats['max_e'], dtype=torch.float32),
        'min_c': torch.tensor(stats['min_c'], dtype=torch.float32),
        'max_c': torch.tensor(stats['max_c'], dtype=torch.float32)
    }

# ---------------- Prediction API ----------------
def predict_batch_from_features(X_features_np, norm='per_sample'):
    """
    Input:
        X_features_np: numpy array shape (N, F) -- PREPROCESSED exactly as training.
    Returns:
        dict with keys:
          'pred'            : final predicted class indices (np.array)
          'p_final'         : final fused probabilities (np.array NxC)
          'p_proto'         : proto_net probabilities (np.array NxC)
          'p_xgb'           : xgboost probabilities (np.array NxC) or None
          'accepted'        : boolean mask (np.array)
          'tier'            : array of 'known'/'rare'/'unknown'
          'proto_conf'      : max proto probability per sample
          'dmin'            : min fused distance per sample
          'explain'         : short reason strings per sample
    """
    model_device = device
    # convert features to torch and compute embedding (on device)
    X = torch.tensor(X_features_np.astype(np.float32), device=model_device)
    with torch.no_grad():
        emb = model(X)  # (N, z) on device

        # compute fused distances using the project's helper (returned as CPU tensors)
        d_fused_cpu, d_e_cpu, d_c_cpu = _compute_fused(emb.detach().cpu(), centroids.detach().cpu(), model, norm=norm, global_stats=global_stats)

        # proto logits & probs (use model.tau_param if present)
        if hasattr(model, 'tau_param'):
            try:
                tau_model = (F.softplus(model.tau_param).item() + 1e-6)
            except Exception:
                tau_model = 1.0
        else:
            tau_model = 1.0

        logits_proto = - d_fused_cpu.to(model_device) / tau_model
        p_proto = F.softmax(logits_proto, dim=1).detach().cpu().numpy()  # (N, C)

        dmin = d_fused_cpu.min(dim=1).values.numpy()  # lower = closer
        proto_conf = p_proto.max(axis=1)

        # gating: accept only samples that satisfy both criteria
        accepted_mask = (dmin <= tau_dist) & (proto_conf >= p_thresh)

        # XGBoost on embeddings: if booster available
        if _have_xgb:
            emb_cpu = emb.detach().cpu().numpy()
            dmat = xgb.DMatrix(emb_cpu)
            p_xgb = _booster.predict(dmat)  # (N, C)
        else:
            p_xgb = None

        # hybrid fusion
        if p_xgb is not None:
            p_final = best_w * p_proto + (1.0 - best_w) * p_xgb
        else:
            p_final = p_proto.copy()

        pred = p_final.argmax(axis=1)
        final_conf = p_final.max(axis=1)

        # tiering/explanation logic (matches your Cell 11 defaults)
        LOW_CONF_THRESH = 0.50
        HIGH_CONF_THRESH = 0.80
        KNOWN_MARGIN = 0.30
        RARE_MARGIN = 0.60

        tiers = []
        explanations = []
        N = p_final.shape[0]
        for i in range(N):
            if not accepted_mask[i]:
                tiers.append("unknown")
                explanations.append("rejected_by_gate")
                continue

            fused_score = float(final_conf[i])
            dm = float(dmin[i])
            proto_arg = int(p_proto[i].argmax()) if isinstance(p_proto, np.ndarray) else int(torch.tensor(p_proto[i]).argmax().item())
            xgb_arg = int(p_xgb[i].argmax()) if (p_xgb is not None) else proto_arg
            fused_arg = int(pred[i])

            reasons = []
            if proto_arg != xgb_arg:
                reasons.append("model_disagreement")
            if fused_score >= HIGH_CONF_THRESH and dm <= KNOWN_MARGIN:
                reasons.append("high_confidence")
                tiers.append("known")
            elif fused_score < LOW_CONF_THRESH or dm > RARE_MARGIN:
                reasons.append("low_confidence_or_far")
                tiers.append("rare")
            else:
                reasons.append("near_margin")
                tiers.append("rare")
            explanations.append(",".join(reasons))

        return {
            'pred': pred,
            'p_final': p_final,
            'p_proto': p_proto,
            'p_xgb': p_xgb,
            'accepted': accepted_mask,
            'tier': np.array(tiers),
            'proto_conf': proto_conf,
            'dmin': dmin,
            'explain': np.array(explanations)
        }

# ---------- Example usage ----------
# Ensure your input features are preprocessed exactly like training:
# X_test_feats = some_preprocessed_numpy_array(shape=(M, F))
# out = predict_batch_from_features(X_test_feats, norm='per_sample')
# print("predictions:", out['pred'][:10], "tiers:", out['tier'][:10], "accepted rate:", out['accepted'].mean())
# Save or return 'out' as needed for your portal / API.

# ------------------ End of Prediction Module ------------------
