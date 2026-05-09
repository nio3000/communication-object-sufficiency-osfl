import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset,TensorDataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

from src.data.datasets import DATASET_LOADERS, get_targets
from src.data.partition import dirichlet_label_partition, quantity_skew_partition, mixed_partition
from src.fl.aggregation import (
    aggregate_classwise_statistics,
    adaptive_weights,
    build_statistical_head,
    default_head_for_object_level,
    estimate_message_scalars,
    hierarchical_conflict_suppressed_aggregation,
    infer_head_backbone_keys,
    rgca_aggregation,
    select_hamoc_object_level,
)
from src.fl.algorithms import reptile_meta_init
from src.fl.client import FLClient
from src.fl.server import FLServer
from src.utils.seed import set_seed


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def get_device(cfg):
    hw = cfg.get("hardware", {})
    mode = hw.get("device", "auto")
    gpu_id = hw.get("gpu_id", 0)
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1 and y.shape[-1] == 1:
                y = y.squeeze(-1).long()
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(labels, preds)
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "macro_f1": macro_f1, "balanced_acc": balanced_acc, "auc": auc}


def apply_optional_calibration(server, val_ds, device, cfg):
    calib_cfg = cfg["method"].get("calibration", {})
    if not calib_cfg.get("enabled", False):
        return
    print("🔧 Applying optional server-side head calibration...")
    calib_size = int(calib_cfg.get("calib_size", 500))
    calib_epochs = int(calib_cfg.get("calib_epochs", 10))
    calib_lr = float(calib_cfg.get("lr", 0.002))
    calib_indices = np.random.choice(len(val_ds), min(calib_size, len(val_ds)), replace=False)
    calib_loader = DataLoader(Subset(val_ds, calib_indices), batch_size=32, shuffle=True)
    model_to_calib = server.global_model.to(device)
    trainable_params = []
    target_keywords = ["fc", "classifier", "linear", "out", "head"]
    for name, param in model_to_calib.named_parameters():
        if any(k in name.lower() for k in target_keywords):
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = False
    if not trainable_params:
        all_params = list(model_to_calib.parameters())
        for p in all_params[-4:]:
            p.requires_grad = True
        trainable_params = all_params[-4:]
    optimizer = torch.optim.Adam(trainable_params, lr=calib_lr)
    criterion = torch.nn.CrossEntropyLoss()
    model_to_calib.train()
    for epoch in range(calib_epochs):
        losses = []
        for x, y in calib_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() > 1:
                y = y.squeeze(1).long()
            optimizer.zero_grad()
            outputs = model_to_calib(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"   [Calib {epoch+1:02d}/{calib_epochs}] loss={np.mean(losses):.4f}")
    for p in model_to_calib.parameters():
        p.requires_grad = True
    server.set_state(model_to_calib.state_dict())

def _stable_cholesky(cov: torch.Tensor, jitter: float = 1e-4) -> torch.Tensor:
    cov = 0.5 * (cov + cov.T)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cur_jitter = jitter
    for _ in range(6):
        try:
            return torch.linalg.cholesky(cov + cur_jitter * eye)
        except RuntimeError:
            cur_jitter *= 10.0
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=jitter)
    return eigvecs @ torch.diag(torch.sqrt(eigvals))


def _build_o4_virtual_dataset(
    global_stats,
    device,
    samples_per_class: int = 200,
    cov_scale: float = 1.0,
    jitter: float = 1e-4,
    balance_mode: str = "empirical",
):
    means = torch.tensor(global_stats["means"], dtype=torch.float32, device=device)
    priors = torch.tensor(global_stats["priors"], dtype=torch.float32, device=device)
    pooled_cov = torch.tensor(global_stats["pooled_cov"], dtype=torch.float32, device=device)

    num_classes, feature_dim = means.shape
    pooled_cov = 0.5 * (pooled_cov + pooled_cov.T)
    pooled_cov = cov_scale * pooled_cov
    chol = _stable_cholesky(pooled_cov, jitter=jitter)

    if balance_mode == "equal":
        counts = [samples_per_class] * num_classes
    else:
        total = samples_per_class * num_classes
        raw = torch.round(priors * total).long()
        counts = raw.tolist()
        diff = total - sum(counts)
        if diff != 0:
            counts[int(torch.argmax(priors).item())] += diff
        counts = [max(1, c) for c in counts]

    xs, ys = [], []
    for c in range(num_classes):
        n_c = int(counts[c])
        noise = torch.randn(n_c, feature_dim, device=device)
        z = noise @ chol.T + means[c].unsqueeze(0)
        y = torch.full((n_c,), c, dtype=torch.long, device=device)
        xs.append(z)
        ys.append(y)

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return TensorDataset(X, Y)


def _base_head_predict_logits(base_head, features: torch.Tensor) -> torch.Tensor:
    if hasattr(base_head, "predict_logits"):
        return base_head.predict_logits(features)

    head_type = getattr(base_head, "head_type", None)
    if head_type == "lda":
        W = torch.tensor(base_head.params["W"], dtype=features.dtype, device=features.device)
        b = torch.tensor(base_head.params["b"], dtype=features.dtype, device=features.device)
        return features @ W.T + b

    raise ValueError(f"Unsupported base head for residual refinement: {head_type}")


class O4ResidualRefinedHead:
    def __init__(self, base_head, residual_head: torch.nn.Module, residual_scale: float):
        self.base_head = base_head
        self.residual_head = residual_head.eval()
        self.residual_scale = float(residual_scale)

        self.head_type = "o4_refine_v2"
        self.base_head_type = getattr(base_head, "head_type", "unknown")
        self.num_classes = residual_head.out_features
        self.feature_dim = residual_head.in_features
        self.params = {
            "base_head_type": self.base_head_type,
            "residual_scale": self.residual_scale,
            "residual_weight": residual_head.weight.detach().cpu(),
            "residual_bias": residual_head.bias.detach().cpu(),
        }

    @torch.no_grad()
    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        base_logits = _base_head_predict_logits(self.base_head, features)
        residual_logits = self.residual_head(features)
        return base_logits + self.residual_scale * residual_logits


def apply_o4_refine_v2(
    server,
    global_stats,
    base_head,
    device,
    refine_cfg,
    num_classes: int,
    feature_dim: int,
):
    if not refine_cfg.get("enabled", False):
        return base_head

    print("🔥 Applying O4-Refine-v2 (residual linear correction around LDA)...")

    samples_per_class = int(refine_cfg.get("samples_per_class", 200))
    epochs = int(refine_cfg.get("epochs", 5))
    lr = float(refine_cfg.get("lr", 0.005))
    batch_size = int(refine_cfg.get("batch_size", 512))
    optimizer_name = str(refine_cfg.get("optimizer", "sgd")).lower()
    scheduler_name = str(refine_cfg.get("scheduler", "cosine")).lower()
    weight_decay = float(refine_cfg.get("weight_decay", 1e-3))
    label_smoothing = float(refine_cfg.get("label_smoothing", 0.0))
    cov_scale = float(refine_cfg.get("cov_scale", 1.0))
    jitter = float(refine_cfg.get("jitter", 1e-4))
    balance_mode = str(refine_cfg.get("balance_mode", "empirical"))
    residual_scale = float(refine_cfg.get("residual_scale", 0.1))
    residual_l2 = float(refine_cfg.get("residual_l2", 1.0))
    init_zero_residual = bool(refine_cfg.get("init_zero_residual", True))

    virtual_ds = _build_o4_virtual_dataset(
        global_stats=global_stats,
        device=device,
        samples_per_class=samples_per_class,
        cov_scale=cov_scale,
        jitter=jitter,
        balance_mode=balance_mode,
    )
    virtual_loader = DataLoader(virtual_ds, batch_size=batch_size, shuffle=True)

    residual_head = torch.nn.Linear(feature_dim, num_classes).to(device)
    if init_zero_residual:
        with torch.no_grad():
            residual_head.weight.zero_()
            residual_head.bias.zero_()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            residual_head.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            residual_head.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    residual_head.train()
    for epoch in range(epochs):
        losses = []
        for batch_x, batch_y in virtual_loader:
            optimizer.zero_grad()

            with torch.no_grad():
                base_logits = _base_head_predict_logits(base_head, batch_x)

            residual_logits = residual_head(batch_x)
            logits = base_logits + residual_scale * residual_logits

            ce_loss = criterion(logits, batch_y)
            reg_loss = residual_l2 * (
                residual_head.weight.pow(2).mean() + residual_head.bias.pow(2).mean()
            )
            loss = ce_loss + reg_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            print(f"   [O4-Refine-v2 {epoch+1:02d}/{epochs}] loss={np.mean(losses):.4f}")

    refined_head = O4ResidualRefinedHead(
        base_head=base_head,
        residual_head=residual_head,
        residual_scale=residual_scale,
    )
    server.set_external_head(refined_head)
    print("✅ O4-Refine-v2 completed.")
    print("residual_weight_norm =", residual_head.weight.norm().item())
    print("residual_bias_norm =", residual_head.bias.norm().item())
    return refined_head

def _load_dataset_and_model(cfg):
    dataset_name = cfg["dataset"]["name"].lower()
    train_ds, val_ds, test_ds = DATASET_LOADERS[dataset_name](cfg["dataset"]["data_root"])

    # num_classes 优先从配置读取；否则才尝试从 MedMNIST info 中读取
    if "num_classes" in cfg["dataset"]:
        num_classes = int(cfg["dataset"]["num_classes"])
    elif hasattr(train_ds, "info") and "label" in train_ds.info:
        num_classes = int(len(train_ds.info["label"]))
    else:
        raise ValueError(
            f"num_classes is required for dataset={dataset_name}. "
            "Please set dataset.num_classes in yaml."
        )

    # in_channels 优先从配置读取；否则兼容 MedMNIST；CIFAR-10 默认 3
    if "in_channels" in cfg["dataset"]:
        in_channels = int(cfg["dataset"]["in_channels"])
    elif hasattr(train_ds, "info") and "n_channels" in train_ds.info:
        in_channels = int(train_ds.info["n_channels"])
    elif dataset_name in ["cifar10","cifar100"]:
        in_channels = 3
    else:
        raise ValueError(
            f"in_channels is required for dataset={dataset_name}. "
            "Please set dataset.in_channels in yaml."
        )

    model_name = cfg.get("model", {}).get("name", "small_cnn").lower()

    if model_name == "resnet18":
        from src.models.resnet import ResNet18ForMedMNIST as ModelClass
    else:
        from src.models.cnn import SmallCNN as ModelClass

    def create_model():
        return ModelClass(num_classes=num_classes, in_channels=in_channels)

    return train_ds, val_ds, test_ds, num_classes, create_model
def _ensure_non_empty_clients(client_indices, cfg, seed):
    """
    Repair empty clients after severe Dirichlet label-skew partitioning.

    In very small alpha settings, e.g. alpha=0.05, Dirichlet partitioning may
    assign zero samples to some clients. PyTorch RandomSampler cannot build a
    DataLoader from an empty dataset. To keep the number of clients fixed, we
    minimally move samples from the largest clients to empty clients.

    This should be applied consistently to all methods in the severity scan.
    """
    import numpy as np

    non_iid_cfg = cfg.get("federated", {}).get("non_iid", {})
    min_samples = int(non_iid_cfg.get("min_samples_per_client", 1))

    if min_samples <= 0:
        return client_indices

    rng = np.random.default_rng(int(seed) + 20260428)

    fixed = []
    for idx in client_indices:
        arr = np.asarray(idx).reshape(-1)
        fixed.append([int(x) for x in arr.tolist()])

    n_clients = len(fixed)

    for cid in range(n_clients):
        while len(fixed[cid]) < min_samples:
            donors = [
                j for j in range(n_clients)
                if len(fixed[j]) > min_samples
            ]

            if not donors:
                raise RuntimeError(
                    f"Cannot repair empty clients: no donor has more than "
                    f"{min_samples} samples."
                )

            donor = max(donors, key=lambda j: len(fixed[j]))
            move_pos = int(rng.integers(0, len(fixed[donor])))
            moved_sample = fixed[donor].pop(move_pos)
            fixed[cid].append(moved_sample)

            print(
                f"⚠️ Repaired empty/too-small client {cid}: "
                f"moved 1 sample from client {donor}."
            )

    return [np.asarray(idx, dtype=np.int64) for idx in fixed]

def _partition_dataset(cfg, train_ds):
    labels = np.asarray(get_targets(train_ds)).reshape(-1).astype(int)
    skew_type = cfg["federated"]["non_iid"].get("type", "label")
    verbose = cfg.get("monitoring", {}).get("verbose_partition", False)
    seed = int(cfg.get("seed", 42))
    num_clients = int(cfg["federated"]["num_clients"])

    if skew_type == "label":
        client_indices, partition_stats = dirichlet_label_partition(
            labels,
            num_clients=num_clients,
            alpha=cfg["federated"]["non_iid"]["alpha"],
            seed=seed,
            return_stats=True,
            verbose=verbose,
        )

    elif skew_type == "quantity":
        client_indices, partition_stats = quantity_skew_partition(
            labels,
            num_clients=num_clients,
            alpha_q=cfg["federated"]["non_iid"]["alpha_q"],
            seed=seed,
            return_stats=True,
            verbose=verbose,
        )

    elif skew_type == "mixed":
        client_indices, partition_stats = mixed_partition(
            labels,
            num_clients=num_clients,
            alpha_label=cfg["federated"]["non_iid"]["alpha_label"],
            alpha_q=cfg["federated"]["non_iid"]["alpha_q"],
            seed=seed,
            return_stats=True,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown skew type: {skew_type}")

    # ===== 新增：修复 alpha 很小时出现的空客户端 =====
    before_sizes = [len(idx) for idx in client_indices]
    before_empty = sum(1 for s in before_sizes if s == 0)

    client_indices = _ensure_non_empty_clients(client_indices, cfg, seed)

    after_sizes = [len(idx) for idx in client_indices]
    after_empty = sum(1 for s in after_sizes if s == 0)

    if before_empty > 0 or after_empty > 0:
        print(
            f"🧩 Empty-client repair summary: "
            f"before_empty={before_empty}, after_empty={after_empty}, "
            f"min_before={min(before_sizes)}, min_after={min(after_sizes)}"
        )

    # 把修复信息写进 partition_stats，方便后面论文/日志追溯
    if isinstance(partition_stats, dict):
        partition_stats["empty_client_repair"] = {
            "enabled": True,
            "min_samples_per_client": int(
                cfg.get("federated", {})
                   .get("non_iid", {})
                   .get("min_samples_per_client", 1)
            ),
            "before_empty_clients": int(before_empty),
            "after_empty_clients": int(after_empty),
            "before_min_samples": int(min(before_sizes)),
            "after_min_samples": int(min(after_sizes)),
            "before_max_samples": int(max(before_sizes)),
            "after_max_samples": int(max(after_sizes)),
        }

    return client_indices, partition_stats

def _create_clients(cfg, create_model, train_ds, client_indices, device):
    aug_cfg = cfg["train"].get("data_augmentation", {"enabled": False})
    return [
        FLClient(i, create_model, train_ds, client_indices[i], device, batch_size=cfg["train"]["batch_size"], augmentation_config=aug_cfg)
        for i in range(cfg["federated"]["num_clients"])
    ]


def _maybe_meta_init(cfg, create_model, clients, server, device):
    if cfg["method"].get("meta_init", {}).get("enabled", False):
        micfg = cfg["method"]["meta_init"]
        print(f"🛠️ Meta-init enabled: steps={micfg['meta_steps']}")
        meta_state = reptile_meta_init(
            create_model,
            clients,
            device,
            meta_steps=micfg["meta_steps"],
            meta_lr=micfg["meta_lr"],
            inner_steps=micfg["inner_steps"],
            inner_lr=micfg["inner_lr"],
            verbose=True,
        )
        server.set_state(meta_state)


def _save_run_artifacts(run_dir: Path, cfg, partition_stats, payloads: Dict[str, Any]):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(cfg), f, ensure_ascii=False, indent=2)
    with open(run_dir / "partition_stats.json", "w", encoding="utf-8") as f:
        json.dump(_to_serializable(partition_stats), f, ensure_ascii=False, indent=2)
    for name, value in payloads.items():
        path = run_dir / name
        if name.endswith(".csv"):
            pd.DataFrame(value).to_csv(path, index=False)
        elif name.endswith(".pt"):
            torch.save(value, path)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(_to_serializable(value), f, ensure_ascii=False, indent=2)

def _pack_statistical_head(head):
    if head is None:
        return None
    return {
        "head_type": getattr(head, "head_type", None),
        "num_classes": getattr(head, "num_classes", None),
        "feature_dim": getattr(head, "feature_dim", None),
        "params": getattr(head, "params", None),
    }

def _apply_prior_temperature(global_stats, gamma: float = 1.0, eps: float = 1e-12):
    if gamma is None:
        gamma = 1.0
    gamma = float(gamma)

    # gamma = 1.0 时直接返回原对象
    if abs(gamma - 1.0) < 1e-12:
        return global_stats

    stats = copy.deepcopy(global_stats)
    priors = np.asarray(stats["priors"], dtype=np.float64)
    priors = np.clip(priors, eps, 1.0)

    # π_c^gamma，再归一化
    tempered = np.exp(gamma * np.log(priors))
    tempered = tempered / tempered.sum()

    stats["priors"] = tempered
    return stats

def run_parameter_once(cfg, run_idx, seed):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    print(f"🚀 Run {run_idx + 1} | seed={cfg['seed']} | device={device} | exp={cfg['exp_name']}")

    train_ds, val_ds, test_ds, num_classes, create_model = _load_dataset_and_model(cfg)
    client_indices, partition_stats = _partition_dataset(cfg, train_ds)
    clients = _create_clients(cfg, create_model, train_ds, client_indices, device)
    server = FLServer(create_model, device)
    _maybe_meta_init(cfg, create_model, clients, server, device)

    global_w0 = server.get_state()
    init_metrics = evaluate_model(server.global_model, DataLoader(test_ds, batch_size=cfg["train"]["batch_size"]), device, num_classes)
    init_acc = init_metrics["accuracy"]
    print(f"📍 Init accuracy: {init_acc:.4f} | macro_f1: {init_metrics['macro_f1']:.4f}")

    local_updates, client_metas = [], []
    prox_cfg = cfg["method"].get("local_train", {})
    optimizer_type = cfg["train"].get("optimizer", "adamw").lower()
    momentum = cfg["train"].get("momentum", 0.9)
    train_encoder = bool(cfg.get("representation", {}).get("train_encoder", True))
    train_head_only = bool(prox_cfg.get("train_head_only", False))

    for i, client in enumerate(clients):
        trained_w, meta = client.local_train(
            copy.deepcopy(global_w0),
            epochs=cfg["train"]["local_epochs"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"].get("weight_decay", 1e-4),
            prox_mu=float(prox_cfg.get("prox_mu", 0.01)),
            proximal_enabled=bool(prox_cfg.get("proximal_enabled", True)),
            optimizer_type=optimizer_type,
            momentum=momentum,
            label_smoothing=cfg["train"].get("label_smoothing", 0.0),
            train_encoder=train_encoder,
            train_head_only=train_head_only,
            freeze_backbone_bn=bool(cfg.get("representation", {}).get("freeze_backbone_bn", True)),
        )
        delta = {}
        total_norm = 0.0
        for k in global_w0.keys():
            if torch.is_tensor(global_w0[k]) and global_w0[k].dtype.is_floating_point:
                diff = trained_w[k].float().cpu() - global_w0[k].float().cpu()
                delta[k] = diff
                total_norm += torch.norm(diff).item()
            else:
                delta[k] = trained_w[k]
        if i % 5 == 0 or i == len(clients) - 1:
            print(f"   [Client {i:02d}] delta_norm={total_norm:.6f} | entropy={meta['label_entropy']:.4f}")
        local_updates.append(delta)
        client_metas.append(meta)

    agg_details, agg_summary = [], {}
    method_cfg = cfg["method"]
    if method_cfg.get("rgca", {}).get("enabled", False):
        updated_state, agg_details, agg_summary = rgca_aggregation(global_w0, local_updates, client_metas, device=device, return_details=True)
    elif method_cfg.get("hcsa", {}).get("enabled", False):
        updated_state, agg_details, agg_summary = hierarchical_conflict_suppressed_aggregation(global_w0, local_updates, client_metas, device=device, return_all_cosines=True)
    else:
        print("⚖️ Running sample-size weighted FedAvg aggregation...")
        ws = np.array([m["num_samples"] for m in client_metas], dtype=np.float64)
        weights = ws / ws.sum()
        weights_t = torch.tensor(weights, device=device).float()
        updated_state = {}
        agg_lr = float(method_cfg.get("fedavg", {}).get("agg_lr", 1.0))
        for k in global_w0.keys():
            if torch.is_tensor(global_w0[k]) and global_w0[k].dtype.is_floating_point:
                total = torch.zeros_like(global_w0[k], device=device).float()
                for i in range(len(local_updates)):
                    total += weights_t[i] * local_updates[i][k].to(device).float()
                updated_state[k] = global_w0[k].float().to(device) + agg_lr * total
            else:
                updated_state[k] = global_w0[k].to(device)
        agg_details = [{"cid": m["cid"], "num_samples": m["num_samples"], "whole_model_weight": float(weights[i])} for i, m in enumerate(client_metas)]
        agg_summary = {"mode": "fedavg", "agg_lr": agg_lr}

    server.set_state(updated_state)
    apply_optional_calibration(server, val_ds, device, cfg)
    final_metrics = evaluate_model(server.global_model, DataLoader(test_ds, batch_size=cfg["train"]["batch_size"]), device, num_classes)
    final_acc = final_metrics["accuracy"]
    print(f"🔥 Final accuracy: {final_acc:.4f} | gain={(final_acc - init_acc) * 100:.2f}%")

    out_root = Path(cfg.get("output_dir", "outputs"))
    run_dir = out_root / f"{cfg['exp_name']}_seed{cfg['seed']}_run{run_idx+1}_{time.strftime('%Y%m%d-%H%M%S')}"
    _save_run_artifacts(run_dir, cfg, partition_stats, {
        "aggregation_summary.json": agg_summary,
        "aggregation_details.csv": agg_details,
        "global_model.pt": server.global_model.state_dict(),
    })

    return {
        "exp_name": cfg["exp_name"],
        "seed": int(cfg["seed"]),
        "method": cfg["method"].get("name", "unknown"),
        "family": "parameter",
        "final_acc": final_acc,
        "macro_f1": final_metrics["macro_f1"],
        "balanced_acc": final_metrics["balanced_acc"],
        "auc": final_metrics["auc"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
    }



def run_msco_once(cfg, run_idx, seed):
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = int(seed)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    print(f"🚀 Run {run_idx + 1} | seed={cfg['seed']} | device={device} | exp={cfg['exp_name']} [MSCO]")

    train_ds, val_ds, test_ds, num_classes, create_model = _load_dataset_and_model(cfg)
    client_indices, partition_stats = _partition_dataset(cfg, train_ds)
    
    clients = _create_clients(cfg, create_model, train_ds, client_indices, device)
    server = FLServer(create_model, device)
    _maybe_meta_init(cfg, create_model, clients, server, device)

    global_state = server.get_state()
    method_cfg = cfg["method"].get("msco", {})
    object_level = method_cfg.get("object_level", "O4")
    if object_level == "AUTO":
        quick_stats = [
            c.extract_classwise_statistics(global_state, object_level="O2", num_classes=num_classes)
            for c in clients
        ]
        _, diag = select_hamoc_object_level(quick_stats)
        object_level, hamoc_diag = select_hamoc_object_level(quick_stats)
    else:
        hamoc_diag = None
    head_type = method_cfg.get("head_type") or default_head_for_object_level(object_level)

    representation_cfg = cfg.get("representation", {})
    local_adaptation = representation_cfg.get("local_adaptation", "none")
    extract_after_local_train = bool(representation_cfg.get("extract_after_local_train", False))
    train_encoder = local_adaptation == "full"
    train_head_only = local_adaptation == "head_only"

    client_payloads = []
    if extract_after_local_train and local_adaptation != "none":
        prox_cfg = cfg["method"].get("local_train", {})
        optimizer_type = cfg["train"].get("optimizer", "adamw").lower()
        momentum = cfg["train"].get("momentum", 0.9)
        for client in clients:
            trained_w, _ = client.local_train(
                copy.deepcopy(global_state),
                epochs=cfg["train"]["local_epochs"],
                lr=cfg["train"]["lr"],
                weight_decay=cfg["train"].get("weight_decay", 1e-4),
                prox_mu=float(prox_cfg.get("prox_mu", 0.01)),
                proximal_enabled=bool(prox_cfg.get("proximal_enabled", True)),
                optimizer_type=optimizer_type,
                momentum=momentum,
                label_smoothing=cfg["train"].get("label_smoothing", 0.0),
                train_encoder=train_encoder,
                train_head_only=train_head_only,
                freeze_backbone_bn=bool(representation_cfg.get("freeze_backbone_bn", True)),
            )
            client_payloads.append(
                client.extract_classwise_statistics(
                    trained_w,
                    object_level=object_level,
                    max_samples=method_cfg.get("max_stats_samples"),
                    num_classes=num_classes,
                )
            )
    else:
        for client in clients:
            client_payloads.append(
                client.extract_classwise_statistics(
                    global_state,
                    object_level=object_level,
                    max_samples=method_cfg.get("max_stats_samples"),
                    num_classes=num_classes,
                )
            )
    global_stats = aggregate_classwise_statistics(
        client_payloads,
        object_level=object_level,
        eps=float(method_cfg.get("eps", 1e-6)),
    )

    prior_temperature = float(method_cfg.get("prior_temperature", 1.0))
    head_stats = _apply_prior_temperature(
        global_stats,
        gamma=prior_temperature,
        eps=float(method_cfg.get("eps", 1e-6)),
    )

    head = build_statistical_head(
        head_stats,
        head_type=head_type,
        shrinkage=float(method_cfg.get("shrinkage", 1e-3)),
        diag_floor=float(method_cfg.get("diag_floor", 1e-4)),
        eps=float(method_cfg.get("eps", 1e-6)),
        min_class_count_for_full_cov=int(method_cfg.get("min_class_count_for_full_cov", 24)),
        pooled_blend_strength=float(method_cfg.get("pooled_blend_strength", 0.75)),
    )
    server.set_external_head(head)

    refine_cfg = method_cfg.get("refine", {})

    if object_level == "O4" and head_type == "lda" and refine_cfg.get("enabled", False):
        method_name = str(refine_cfg.get("method", "residual_linear")).lower()
        if method_name == "residual_linear":
            head = apply_o4_refine_v2(
                server=server,
                global_stats=global_stats,
                base_head=head,
                device=device,
                refine_cfg=refine_cfg,
                num_classes=num_classes,
                feature_dim=int(global_stats["feature_dim"]),
            )
            print("after refine head type =", getattr(head, "head_type", "unknown"))
    # 执行测试
    metrics = server.test_with_external_head(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        num_classes=num_classes,
    )

    msg = estimate_message_scalars(global_stats, object_level=object_level)
    # print(f"📦 Object={object_level} | Head={head_type} | accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}")
    print(f"📦 Object={object_level} | Head={getattr(head, 'head_type', head_type)} | accuracy={metrics['accuracy']:.4f} | macro_f1={metrics['macro_f1']:.4f}")
    
    global_summary = {
        "object_level": object_level,
        "head_type": getattr(head, "head_type", head_type),
        "num_classes": int(global_stats["num_classes"]),
        "feature_dim": int(global_stats["feature_dim"]),
        "counts": global_stats["counts"].tolist(),
        "priors_raw": global_stats["priors"].tolist(),
        "prior_temperature": prior_temperature,
        "priors_effective": head_stats["priors"].tolist(),
        "message_size": msg,
        "hamoc_diagnostics": hamoc_diag,
        "client_diagnostics": global_stats["client_diagnostics"],
        "refine": method_cfg.get("refine", {"enabled": False}),
    }
    if global_stats.get("pooled_cov") is not None:
        pooled_diag = np.diag(global_stats["pooled_cov"])
        global_summary["pooled_cov_trace"] = float(np.trace(global_stats["pooled_cov"]))
        global_summary["pooled_cov_diag_mean"] = float(pooled_diag.mean())
    if head_type == "qda":
        global_summary["qda_fallback_mask"] = head.params["fallback_mask"].tolist()
        global_summary["qda_effective_blends"] = head.params["effective_blends"].tolist()

    out_root = Path(cfg.get("output_dir", "outputs"))
    run_dir = out_root / f"{cfg['exp_name']}_seed{cfg['seed']}_run{run_idx+1}_{time.strftime('%Y%m%d-%H%M%S')}"
    head_pt = _pack_statistical_head(head)
    msco_bundle = {
        "config": cfg,
        "partition_stats": partition_stats,
        "metrics": metrics,
        "message_size": msg,
        "global_stats": global_stats,
        "global_summary": global_summary,
        "head": head_pt,
        "seed": int(cfg["seed"]),
        "exp_name": cfg["exp_name"],
        "run_idx": int(run_idx + 1),
    }

    _save_run_artifacts(run_dir, cfg, partition_stats, {
        "global_stats_summary.json": global_summary,
        "client_stats_details.csv": global_stats["client_diagnostics"],

        # MSCO 按 seed/run 目录保存的 pt
        "global_model.pt": server.global_model.state_dict(),
        "statistical_head.pt": head_pt,
        "global_stats.pt": global_stats,
        "msco_bundle.pt": msco_bundle,
    })

    return {
        "exp_name": cfg["exp_name"],
        "seed": int(cfg["seed"]),
        "method": cfg["method"].get("name", "unknown"),
        "family": "msco",
        "object_level": object_level,
        "head_type": getattr(head, "head_type", head_type),
        "message_scalars": int(msg["message_scalars"]),
        "final_acc": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_acc": metrics["balanced_acc"],
        "auc": metrics["auc"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
    }

def run_once(cfg, run_idx, seed):
    family = cfg.get("method", {}).get("family", "parameter")
    if family == "msco":
        return run_msco_once(cfg, run_idx, seed)
    return run_parameter_once(cfg, run_idx, seed)



def main(cfg_path="configs/pathmnist_labelskew_msco_o4_lda.yaml"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seeds = cfg.get("seeds")
    if not seeds:
        base_seed = int(cfg.get("seed", 42))
        num_runs = int(cfg.get("experiment", {}).get("num_runs", 1))
        seeds = [base_seed + 10 * i for i in range(num_runs)]
    all_results = [run_once(cfg, run_idx=i, seed=seed) for i, seed in enumerate(seeds)]
    out_root = Path(cfg.get("output_dir", "outputs"))
    out_root.mkdir(parents=True, exist_ok=True)
    results_path = out_root / "results.csv"
    summary_path = out_root / "summary.json"
    df = pd.DataFrame(all_results)
    df.to_csv(results_path, mode="a", header=not results_path.exists(), index=False)
    summary = {
        "exp_name": cfg["exp_name"],
        "num_runs": len(all_results),
        "all_acc": [float(r["final_acc"]) for r in all_results],
        "mean_acc": float(np.mean([r["final_acc"] for r in all_results])),
        "std_acc": float(np.std([r["final_acc"] for r in all_results])),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(f"✅ Finished {cfg['exp_name']} | mean_acc={summary['mean_acc']:.4f} | std={summary['std_acc']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/pathmnist_labelskew_msco_o4_lda.yaml"
    main(config_file)
