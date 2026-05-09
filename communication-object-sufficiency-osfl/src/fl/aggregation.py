from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


HEAD_KEYWORDS_DEFAULT = ["fc", "classifier", "linear", "head", "out", "proj"]


def _is_float_tensor(x):
    return torch.is_tensor(x) and x.dtype.is_floating_point


def flatten_delta(delta: Dict[str, torch.Tensor], keys: Optional[List[str]] = None, device=None):
    vecs = []
    iterate_keys = keys if keys is not None else sorted(delta.keys())
    for k in iterate_keys:
        if k not in delta:
            continue
        v = delta[k]
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        if torch.is_floating_point(v):
            v = v.reshape(-1).float()
            if device is not None:
                v = v.to(device)
            vecs.append(v)
    if not vecs:
        return torch.zeros(1, device=device)
    return torch.cat(vecs)


def infer_head_backbone_keys(state_dict: Dict[str, torch.Tensor], head_keywords: Optional[List[str]] = None, fallback_last_n: int = 2):
    head_keywords = head_keywords or HEAD_KEYWORDS_DEFAULT
    float_keys = [k for k, v in state_dict.items() if _is_float_tensor(v)]
    head_keys = [k for k in float_keys if any(word in k.lower() for word in head_keywords)]
    if not head_keys:
        head_keys = float_keys[-fallback_last_n:] if len(float_keys) >= fallback_last_n else float_keys[-1:]
    head_keys = sorted(list(dict.fromkeys(head_keys)))
    backbone_keys = [k for k in float_keys if k not in head_keys]
    return backbone_keys, head_keys


def split_delta_groups(delta: Dict[str, torch.Tensor], backbone_keys: List[str], head_keys: List[str]):
    return {
        "backbone": {k: delta[k] for k in backbone_keys if k in delta},
        "head": {k: delta[k] for k in head_keys if k in delta},
    }


def _normalize_np(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    s = float(x.sum())
    if s <= eps:
        return np.ones_like(x) / max(len(x), 1)
    return x / (s + eps)


@torch.no_grad()
def _group_geometry(deltas, keys, device=None, eps=1e-12):
    vecs = [flatten_delta(d, keys=keys, device=device) for d in deltas]
    stacked = torch.stack(vecs, dim=0)
    ref = stacked.mean(dim=0)
    ref_norm = ref / (torch.norm(ref) + eps)
    norms = torch.norm(stacked, dim=1)
    cosine = torch.mv(stacked, ref_norm) / (norms + eps)
    return stacked, ref_norm, cosine, norms


@torch.no_grad()
def _direction_weights(cosine, temperature=0.5, top_k_ratio=0.75):
    num_clients = int(cosine.shape[0])
    num_keep = max(1, int(round(num_clients * top_k_ratio)))
    _, top_idx = torch.topk(cosine, k=num_keep)
    mask = torch.full_like(cosine, float("-inf"))
    mask[top_idx] = 0.0
    weights = F.softmax((cosine + mask) / max(temperature, 1e-6), dim=0)
    selected = torch.zeros_like(cosine, dtype=torch.bool)
    selected[top_idx] = True
    return weights, selected


@torch.no_grad()
def _suppress_conflict(delta_group, ref_direction, beta=0.3, eps=1e-12):
    vec = flatten_delta(delta_group, device=ref_direction.device)
    proj_coeff = torch.dot(vec, ref_direction) / (torch.dot(ref_direction, ref_direction) + eps)
    parallel = proj_coeff * ref_direction
    orth = vec - parallel
    suppressed_vec = vec - beta * orth
    out = {}
    cursor = 0
    for k in sorted(delta_group.keys()):
        v = delta_group[k]
        if not _is_float_tensor(v):
            out[k] = v
            continue
        numel = v.numel()
        out[k] = suppressed_vec[cursor: cursor + numel].reshape_as(v).detach().cpu()
        cursor += numel
    return out


@torch.no_grad()
def hierarchical_conflict_suppressed_aggregation(
    global_state: Dict[str, torch.Tensor],
    deltas: List[Dict[str, torch.Tensor]],
    metas: List[Dict[str, Any]],
    head_keywords: Optional[List[str]] = None,
    backbone_sample_weight: float = 0.35,
    backbone_direction_weight: float = 0.65,
    head_sample_weight: float = 0.20,
    head_entropy_weight: float = 0.20,
    head_direction_weight: float = 0.60,
    backbone_temperature: float = 0.7,
    head_temperature: float = 0.4,
    backbone_top_k_ratio: float = 0.90,
    head_top_k_ratio: float = 0.70,
    backbone_beta: float = 0.15,
    head_beta: float = 0.45,
    agg_lr_backbone: float = 0.80,
    agg_lr_head: float = 0.90,
    eps: float = 1e-12,
    device=None,
    return_all_cosines: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone_keys, head_keys = infer_head_backbone_keys(global_state, head_keywords=head_keywords)
    sample_sizes = np.array([float(m.get("num_samples", 1.0)) for m in metas], dtype=np.float64)
    sample_weights = _normalize_np(sample_sizes, eps=eps)
    entropies = np.array([float(m.get("label_entropy", 0.0)) for m in metas], dtype=np.float64)
    entropies = entropies - entropies.min() + eps
    entropy_weights = _normalize_np(entropies, eps=eps)

    _, backbone_ref, backbone_cos, _ = _group_geometry(deltas, backbone_keys, device=device, eps=eps)
    backbone_dir_w, backbone_selected = _direction_weights(backbone_cos, temperature=backbone_temperature, top_k_ratio=backbone_top_k_ratio)
    backbone_weights = backbone_sample_weight * torch.tensor(sample_weights, device=device, dtype=torch.float32) + backbone_direction_weight * backbone_dir_w
    backbone_weights = backbone_weights / (backbone_weights.sum() + eps)

    _, head_ref, head_cos, _ = _group_geometry(deltas, head_keys, device=device, eps=eps)
    head_dir_w, head_selected = _direction_weights(head_cos, temperature=head_temperature, top_k_ratio=head_top_k_ratio)
    head_weights = (
        head_sample_weight * torch.tensor(sample_weights, device=device, dtype=torch.float32)
        + head_entropy_weight * torch.tensor(entropy_weights, device=device, dtype=torch.float32)
        + head_direction_weight * head_dir_w
    )
    head_weights = head_weights / (head_weights.sum() + eps)

    suppressed_backbone, suppressed_head = [], []
    for d in deltas:
        groups = split_delta_groups(d, backbone_keys, head_keys)
        suppressed_backbone.append(_suppress_conflict(groups["backbone"], backbone_ref, beta=backbone_beta, eps=eps))
        suppressed_head.append(_suppress_conflict(groups["head"], head_ref, beta=head_beta, eps=eps))

    updated_state = {}
    for k, v in global_state.items():
        if not _is_float_tensor(v):
            updated_state[k] = v.detach().clone().to(device)
            continue
        if k in backbone_keys:
            total = torch.zeros_like(v, device=device).float()
            for i in range(len(deltas)):
                if k in suppressed_backbone[i]:
                    total += backbone_weights[i] * suppressed_backbone[i][k].to(device).float()
            updated_state[k] = v.to(device).float() + agg_lr_backbone * total
        elif k in head_keys:
            total = torch.zeros_like(v, device=device).float()
            for i in range(len(deltas)):
                if k in suppressed_head[i]:
                    total += head_weights[i] * suppressed_head[i][k].to(device).float()
            updated_state[k] = v.to(device).float() + agg_lr_head * total
        else:
            updated_state[k] = v.detach().clone().to(device)

    details = []
    for i, meta in enumerate(metas):
        details.append({
            "cid": int(meta.get("cid", i)),
            "num_samples": float(meta.get("num_samples", 0)),
            "label_entropy": float(meta.get("label_entropy", 0.0)),
            "class_coverage": int(meta.get("class_coverage", 0)),
            "backbone_cosine": float(backbone_cos[i].detach().cpu().item()),
            "head_cosine": float(head_cos[i].detach().cpu().item()),
            "backbone_selected_topk": int(backbone_selected[i].detach().cpu().item()),
            "head_selected_topk": int(head_selected[i].detach().cpu().item()),
            "backbone_weight": float(backbone_weights[i].detach().cpu().item()),
            "head_weight": float(head_weights[i].detach().cpu().item()),
        })

    summary = {
        "backbone_keys": backbone_keys,
        "head_keys": head_keys,
        "backbone_mean_cosine": float(backbone_cos.mean().detach().cpu().item()),
        "head_mean_cosine": float(head_cos.mean().detach().cpu().item()),
        "backbone_selected_clients": int(backbone_selected.sum().detach().cpu().item()),
        "head_selected_clients": int(head_selected.sum().detach().cpu().item()),
        "agg_lr_backbone": float(agg_lr_backbone),
        "agg_lr_head": float(agg_lr_head),
        "backbone_beta": float(backbone_beta),
        "head_beta": float(head_beta),
    }
    if return_all_cosines:
        summary["backbone_cosines"] = backbone_cos.detach().cpu().tolist()
        summary["head_cosines"] = head_cos.detach().cpu().tolist()
    return updated_state, details, summary


@torch.no_grad()
def adaptive_weights(deltas, metas, w_sample=0.3, w_direction=0.7, temperature=1.0, top_k_ratio=0.8, eps=1e-12, device=None):
    num_clients = len(deltas)
    if num_clients == 0:
        return np.array([])
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    flattened = [flatten_delta(d, device=device) for d in deltas]
    stacked = torch.stack(flattened)
    ref = stacked.mean(dim=0)
    ref_norm = ref / (torch.norm(ref) + eps)
    norms = torch.norm(stacked, dim=1)
    cos_sim = torch.mv(stacked, ref_norm) / (norms + eps)
    num_keep = max(1, int(num_clients * top_k_ratio))
    _, top_indices = torch.topk(cos_sim, k=num_keep)
    mask = torch.full_like(cos_sim, float("-inf"))
    mask[top_indices] = 0
    direction_weights = F.softmax((cos_sim + mask) / temperature, dim=0)
    sample_sizes = torch.tensor([float(m["num_samples"]) for m in metas], device=device)
    sample_mask = torch.zeros_like(sample_sizes)
    sample_mask[top_indices] = 1.0
    filtered_sample_sizes = sample_sizes * sample_mask
    sample_weights = filtered_sample_sizes / (torch.sum(filtered_sample_sizes) + eps)
    final_weights = w_sample * sample_weights + w_direction * direction_weights
    final_weights = final_weights / (torch.sum(final_weights) + eps)
    return final_weights.detach().cpu().numpy()


@torch.no_grad()
def rgca_aggregation(
    global_state: Dict[str, torch.Tensor],
    deltas: List[Dict[str, torch.Tensor]],
    metas: List[Dict[str, Any]],
    head_keywords: Optional[List[str]] = None,
    tau_b: float = 8.0,
    tau_h: float = 8.0,
    theta_b: float = 0.75,
    theta_h: float = 0.45,
    alpha_b: float = 0.15,
    alpha_h: float = 0.25,
    lambda_e: float = 0.1,
    rho_b: float = 0.1,
    rho_h: float = 0.1,
    eta: float = 1.0,
    eps: float = 1e-12,
    device=None,
    return_details: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha_b = float(max(0.0, min(1.0, alpha_b)))
    alpha_h = float(max(0.0, min(1.0, alpha_h)))
    backbone_keys, head_keys = infer_head_backbone_keys(global_state, head_keywords=head_keywords)
    deltas_b = [{k: d[k] for k in backbone_keys if k in d} for d in deltas]
    deltas_h = [{k: d[k] for k in head_keys if k in d} for d in deltas]
    n = torch.tensor([m.get("num_samples", 1.0) for m in metas], dtype=torch.float32, device=device)
    p = n / (n.sum() + eps)
    flat_b = [flatten_delta(d, device=device) for d in deltas_b]
    flat_h = [flatten_delta(d, device=device) for d in deltas_h]
    stack_b = torch.stack(flat_b, dim=0)
    stack_h = torch.stack(flat_h, dim=0)
    mean_b = torch.sum(p[:, None] * stack_b, dim=0)
    mean_h = torch.sum(p[:, None] * stack_h, dim=0)
    cos_b = torch.stack([F.cosine_similarity(f, mean_b, dim=0, eps=eps) for f in flat_b])
    cos_h = torch.stack([F.cosine_similarity(f, mean_h, dim=0, eps=eps) for f in flat_h])
    e = torch.tensor([m.get("label_entropy", 0.0) for m in metas], dtype=torch.float32, device=device)
    r_b = torch.sigmoid(tau_b * (cos_b - theta_b))
    r_h = torch.sigmoid(tau_h * (cos_h - theta_h)) * torch.exp(-lambda_e * e)
    r_b = rho_b + (1.0 - rho_b) * r_b
    r_h = rho_h + (1.0 - rho_h) * r_h
    w_hat_b = (p * r_b) / ((p * r_b).sum() + eps)
    w_hat_h = (p * r_h) / ((p * r_h).sum() + eps)
    w_b = (1.0 - alpha_b) * p + alpha_b * w_hat_b
    w_h = (1.0 - alpha_h) * p + alpha_h * w_hat_h
    agg_backbone = {k: torch.zeros_like(global_state[k], device=global_state[k].device) for k in backbone_keys if _is_float_tensor(global_state[k])}
    agg_head = {k: torch.zeros_like(global_state[k], device=global_state[k].device) for k in head_keys if _is_float_tensor(global_state[k])}
    for i in range(len(deltas)):
        for k in agg_backbone:
            agg_backbone[k] += w_b[i] * deltas_b[i][k].to(global_state[k].device)
        for k in agg_head:
            agg_head[k] += w_h[i] * deltas_h[i][k].to(global_state[k].device)
    updated_state = {}
    for k, v in global_state.items():
        if not _is_float_tensor(v):
            updated_state[k] = v.detach().clone().to(device)
        elif k in agg_backbone:
            updated_state[k] = v.to(device).float() + eta * agg_backbone[k]
        elif k in agg_head:
            updated_state[k] = v.to(device).float() + eta * agg_head[k]
        else:
            updated_state[k] = v.detach().clone().to(device)
    if not return_details:
        return updated_state
    details = []
    for i, meta in enumerate(metas):
        details.append({
            "cid": int(meta.get("cid", i)),
            "num_samples": float(meta.get("num_samples", 0)),
            "label_entropy": float(meta.get("label_entropy", 0.0)),
            "class_coverage": int(meta.get("class_coverage", 0)),
            "backbone_cosine": float(cos_b[i].detach().cpu().item()),
            "head_cosine": float(cos_h[i].detach().cpu().item()),
            "backbone_reliability": float(r_b[i].detach().cpu().item()),
            "head_reliability": float(r_h[i].detach().cpu().item()),
            "backbone_weight": float(w_b[i].detach().cpu().item()),
            "head_weight": float(w_h[i].detach().cpu().item()),
        })
    summary = {
        "backbone_keys": backbone_keys,
        "head_keys": head_keys,
        "backbone_mean_cosine": float(cos_b.mean().detach().cpu().item()),
        "head_mean_cosine": float(cos_h.mean().detach().cpu().item()),
        "backbone_mean_reliability": float(r_b.mean().detach().cpu().item()),
        "head_mean_reliability": float(r_h.mean().detach().cpu().item()),
        "alpha_b": alpha_b,
        "alpha_h": alpha_h,
        "tau_b": tau_b,
        "tau_h": tau_h,
        "theta_b": theta_b,
        "theta_h": theta_h,
        "lambda_e": lambda_e,
        "rho_b": rho_b,
        "rho_h": rho_h,
        "eta": eta,
    }
    return updated_state, details, summary


# ===== MSCO / statistical-head machinery =====

def _safe_inverse(mat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat + eps * np.eye(mat.shape[0], dtype=mat.dtype))


def _regularize_cov(cov: np.ndarray, shrinkage: float = 1e-3, diag_floor: float = 1e-4) -> np.ndarray:
    d = cov.shape[0]
    tr = float(np.trace(cov))
    target = (tr / max(d, 1)) * np.eye(d, dtype=np.float64)
    out = (1.0 - shrinkage) * cov + shrinkage * target
    diag = np.diag(out).copy()
    diag = np.maximum(diag, diag_floor)
    out[np.diag_indices(d)] = diag
    return out


@dataclass
class StatisticalHead:
    head_type: str
    num_classes: int
    feature_dim: int
    params: Dict[str, Any]

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        x = features.float()
        device = x.device
        priors = torch.as_tensor(self.params.get("priors"), dtype=torch.float32, device=device)
        log_priors = torch.log(priors.clamp_min(1e-12))

        if self.head_type == "prior_only":
            return log_priors.unsqueeze(0).expand(x.size(0), -1)

        means = torch.as_tensor(self.params["means"], dtype=torch.float32, device=device)

        if self.head_type == "ncm":
            x2 = (x ** 2).sum(dim=1, keepdim=True)
            m2 = (means ** 2).sum(dim=1).unsqueeze(0)
            d2 = x2 + m2 - 2.0 * x @ means.t()
            return -0.5 * d2 + log_priors.unsqueeze(0)

        if self.head_type == "diag_gaussian":
            vars_ = torch.as_tensor(self.params["diag_vars"], dtype=torch.float32, device=device)
            diff = x[:, None, :] - means[None, :, :]
            log_det = torch.log(vars_.clamp_min(1e-12)).sum(dim=1)
            quad = (diff * diff / vars_.clamp_min(1e-12)[None, :, :]).sum(dim=2)
            return -0.5 * (quad + log_det[None, :]) + log_priors.unsqueeze(0)

        if self.head_type == "lda":
            W = torch.as_tensor(self.params["W"], dtype=torch.float32, device=device)
            b = torch.as_tensor(self.params["b"], dtype=torch.float32, device=device)
            return x @ W.t() + b.unsqueeze(0)

        if self.head_type == "qda":
            inv_covs = torch.as_tensor(self.params["inv_covs"], dtype=torch.float32, device=device)
            log_dets = torch.as_tensor(self.params["log_dets"], dtype=torch.float32, device=device)
            diff = x[:, None, :] - means[None, :, :]
            quad = torch.einsum("ncd,cde,nce->nc", diff, inv_covs, diff)
            return -0.5 * (quad + log_dets.unsqueeze(0)) + log_priors.unsqueeze(0)

        raise ValueError(f"Unsupported statistical head: {self.head_type}")


def aggregate_classwise_statistics(client_stats_list: List[Dict[str, Any]], object_level: str = "O4", eps: float = 1e-6) -> Dict[str, Any]:
    if len(client_stats_list) == 0:
        raise ValueError("client_stats_list is empty")
    num_classes = int(max(len(cs["counts"]) for cs in client_stats_list))
    feature_dim = int(client_stats_list[0]["feature_dim"])
    counts = np.zeros(num_classes, dtype=np.int64)
    sum_features = np.zeros((num_classes, feature_dim), dtype=np.float64)
    diag_second = np.zeros((num_classes, feature_dim), dtype=np.float64)
    global_second = np.zeros((feature_dim, feature_dim), dtype=np.float64) if object_level in ["O4", "O5"] else None
    class_second = np.zeros((num_classes, feature_dim, feature_dim), dtype=np.float64) if object_level == "O5" else None

    diagnostics = []
    for cs in client_stats_list:
        c = np.asarray(cs["counts"], dtype=np.int64)
        counts[: len(c)] += c
        sf = np.asarray(cs.get("feature_sum", cs.get("sum_features")), dtype=np.float64)
        sum_features[: sf.shape[0], : sf.shape[1]] += sf
        if cs.get("diag_second") is not None:
            ds = np.asarray(cs["diag_second"], dtype=np.float64)
            diag_second[: ds.shape[0], : ds.shape[1]] += ds
        if object_level in ["O4", "O5"] and cs.get("global_second") is not None:
            global_second += np.asarray(cs["global_second"], dtype=np.float64)
        if object_level == "O5" and cs.get("class_second") is not None:
            cs_class_second = np.asarray(cs["class_second"], dtype=np.float64)
            if cs_class_second.shape != (num_classes, feature_dim, feature_dim):
                raise ValueError(
                    f"class_second shape mismatch: got {cs_class_second.shape}, "
                    f"expected {(num_classes, feature_dim, feature_dim)}. "
                    f"Please ensure client.extract_classwise_statistics returns global-class-aligned tensors."
                )
            class_second += cs_class_second
        diagnostics.append({
            "cid": int(cs.get("cid", 0)),
            "num_samples": int(cs.get("num_samples", 0)),
            "class_coverage": int(cs.get("class_coverage", 0)),
            "label_entropy": float(cs.get("label_entropy", 0.0)),
            "min_class_count": int(cs.get("min_class_count", 0)),
            "max_class_count": int(cs.get("max_class_count", 0)),
            "feature_norm_mean": float(cs.get("feature_norm_mean", 0.0)),
            "within_dispersion": float(cs.get("within_dispersion", 0.0)),
            "centroid_overlap": float(cs.get("centroid_overlap", 0.0)),
        })

    means = np.zeros_like(sum_features)
    mask = counts > 0
    means[mask] = sum_features[mask] / counts[mask, None]

    priors = counts.astype(np.float64)
    priors = priors / max(priors.sum(), 1.0)
    priors = np.clip(priors, 1e-12, None)
    priors = priors / priors.sum()

    pooled_cov = None
    if object_level in ["O4", "O5"] and global_second is not None:
        total_n = max(int(counts.sum()), 1)
        centered = global_second.copy()
        for c in range(num_classes):
            if counts[c] > 0:
                centered -= counts[c] * np.outer(means[c], means[c])
        pooled_cov = centered / max(total_n - max(mask.sum(), 1), 1)

    diag_vars = None
    if object_level in ["O3", "O4", "O5"]:
        diag_vars = np.zeros_like(diag_second)
        for c in range(num_classes):
            if counts[c] > 0:
                diag_vars[c] = diag_second[c] / counts[c] - means[c] ** 2

    class_covs = None
    if object_level == "O5" and class_second is not None:
        class_covs = np.zeros_like(class_second)
        for c in range(num_classes):
            if counts[c] > 1:
                centered = class_second[c] - counts[c] * np.outer(means[c], means[c])
                class_covs[c] = centered / max(counts[c] - 1, 1)

    return {
        "object_level": object_level,
        "num_classes": num_classes,
        "feature_dim": feature_dim,
        "counts": counts,
        "priors": priors,
        "means": means,
        "diag_vars": diag_vars,
        "pooled_cov": pooled_cov,
        "class_covs": class_covs,
        "client_diagnostics": diagnostics,
    }


def estimate_message_scalars(global_stats: Dict[str, Any], object_level: str) -> Dict[str, Any]:
    C = int(global_stats["num_classes"])
    D = int(global_stats["feature_dim"])
    if object_level == "O1":
        scalars = C
    elif object_level == "O2":
        scalars = C + C * D
    elif object_level == "O3":
        scalars = C + 2 * C * D
    elif object_level == "O4":
        scalars = C + 2 * C * D + D * D
    elif object_level == "O5":
        scalars = C + 2 * C * D + C * D * D
    else:
        raise ValueError(f"Unknown object_level: {object_level}")
    return {"object_level": object_level, "message_scalars": int(scalars), "feature_dim": D, "num_classes": C}


def default_head_for_object_level(object_level: str) -> str:
    return {"O1": "prior_only", "O2": "ncm", "O3": "diag_gaussian", "O4": "lda", "O5": "qda"}[object_level]


def build_statistical_head(
    global_stats: Dict[str, Any],
    head_type: str,
    shrinkage: float = 1e-3,
    diag_floor: float = 1e-4,
    eps: float = 1e-6,
    min_class_count_for_full_cov: int = 24,
    pooled_blend_strength: float = 0.75,
) -> StatisticalHead:
    head_type = head_type.lower()
    C = int(global_stats["num_classes"])
    D = int(global_stats["feature_dim"])
    priors = np.asarray(global_stats["priors"], dtype=np.float64)
    means = np.asarray(global_stats["means"], dtype=np.float64)
    counts = np.asarray(global_stats["counts"], dtype=np.int64)

    if head_type == "prior_only":
        return StatisticalHead(head_type=head_type, num_classes=C, feature_dim=D, params={"priors": priors})

    if head_type == "ncm":
        return StatisticalHead(head_type=head_type, num_classes=C, feature_dim=D, params={"priors": priors, "means": means})

    if head_type == "diag_gaussian":
        diag_vars = np.asarray(global_stats["diag_vars"], dtype=np.float64)
        diag_vars = np.maximum(diag_vars, diag_floor)
        return StatisticalHead(head_type=head_type, num_classes=C, feature_dim=D, params={"priors": priors, "means": means, "diag_vars": diag_vars})
    if head_type == "lda":
        pooled_cov = np.asarray(global_stats["pooled_cov"], dtype=np.float64)

        # 保证对称
        pooled_cov = 0.5 * (pooled_cov + pooled_cov.T)

        # 正则化
        pooled_cov = _regularize_cov(
            pooled_cov,
            shrinkage=shrinkage,
            diag_floor=diag_floor,
        )

        # 再补 eps 对角，增强数值稳定性
        pooled_cov = pooled_cov + eps * np.eye(D, dtype=np.float64)

        if not np.isfinite(pooled_cov).all():
            raise ValueError("LDA pooled_cov contains non-finite values")

        try:
            W = np.linalg.solve(pooled_cov, means.T).T
        except np.linalg.LinAlgError:
            W = (np.linalg.pinv(pooled_cov) @ means.T).T

        b = -0.5 * np.sum(means * W, axis=1) + np.log(priors)

        if (not np.isfinite(W).all()) or (not np.isfinite(b).all()):
            raise ValueError("LDA head produced non-finite W or b")

        return StatisticalHead(
            head_type=head_type,
            num_classes=C,
            feature_dim=D,
            params={
                "priors": priors,
                "means": means,
                "W": W,
                "b": b,
                "pooled_cov": pooled_cov,
            },
        )

    if head_type == "qda":
        pooled_cov = np.asarray(global_stats["pooled_cov"], dtype=np.float64)
        pooled_cov = _regularize_cov(pooled_cov, shrinkage=shrinkage, diag_floor=diag_floor)
        class_covs = global_stats.get("class_covs")
        if class_covs is None:
            raise ValueError("QDA requires class_covs in global_stats")
        class_covs = np.asarray(class_covs, dtype=np.float64)
        inv_covs = np.zeros_like(class_covs)
        log_dets = np.zeros(C, dtype=np.float64)
        fallback_mask = np.zeros(C, dtype=np.int64)
        effective_blends = np.zeros(C, dtype=np.float64)
        for c in range(C):
            if counts[c] <= 1:
                cov_c = pooled_cov.copy()
                fallback_mask[c] = 1
                effective_blends[c] = 0.0
            else:
                cov_c = class_covs[c]
                lam = min(1.0, counts[c] / max(min_class_count_for_full_cov, 1))
                lam = float(max(0.0, lam))
                lam = pooled_blend_strength * lam
                effective_blends[c] = lam
                cov_c = lam * cov_c + (1.0 - lam) * pooled_cov
                if counts[c] < min_class_count_for_full_cov:
                    fallback_mask[c] = 1
            cov_c = _regularize_cov(cov_c, shrinkage=shrinkage, diag_floor=diag_floor)
            inv_cov = _safe_inverse(cov_c, eps=eps)
            sign, logdet = np.linalg.slogdet(cov_c)
            if sign <= 0:
                cov_c = cov_c + eps * np.eye(D, dtype=np.float64)
                inv_cov = _safe_inverse(cov_c, eps=eps)
                _, logdet = np.linalg.slogdet(cov_c)
            inv_covs[c] = inv_cov
            log_dets[c] = logdet
        return StatisticalHead(
            head_type=head_type,
            num_classes=C,
            feature_dim=D,
            params={
                "priors": priors,
                "means": means,
                "inv_covs": inv_covs,
                "log_dets": log_dets,
                "fallback_mask": fallback_mask,
                "effective_blends": effective_blends,
                "pooled_cov": pooled_cov,
            },
        )

    raise ValueError(f"Unsupported head_type: {head_type}")


def select_hamoc_object_level(client_diagnostics: List[Dict[str, Any]]) -> Tuple[str, Dict[str, float]]:
    if len(client_diagnostics) == 0:
        return "O4", {"score": 0.0}
    entropy = float(np.mean([d.get("label_entropy", 0.0) for d in client_diagnostics]))
    min_class_count = float(np.mean([d.get("min_class_count", 0.0) for d in client_diagnostics]))
    overlap = float(np.mean([d.get("centroid_overlap", 0.0) for d in client_diagnostics]))
    dispersion = float(np.mean([d.get("within_dispersion", 0.0) for d in client_diagnostics]))
    # Simple, interpretable policy: default O4; only choose O5 when class counts are healthy.
    if min_class_count >= 32 and overlap > 0.15:
        level = "O5"
    elif entropy < 0.6 and min_class_count >= 8:
        level = "O3"
    else:
        level = "O4"
    return level, {
        "mean_entropy": entropy,
        "mean_min_class_count": min_class_count,
        "mean_centroid_overlap": overlap,
        "mean_within_dispersion": dispersion,
    }
