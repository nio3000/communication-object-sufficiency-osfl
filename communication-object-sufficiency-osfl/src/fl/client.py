import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch.nn.functional as F

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class FLClient:
    def __init__(
        self,
        cid,
        model_fn,
        train_dataset,
        indices,
        device,
        batch_size: int = 128,
        augmentation_config: Optional[Dict[str, Any]] = None,
    ):
        self.cid = cid
        self.device = device
        self.model_fn = model_fn
        self.batch_size = batch_size
        self.indices = np.asarray(indices).astype(int)

        # 兼容 MedMNIST: imgs / labels
        if hasattr(train_dataset, "imgs") and hasattr(train_dataset, "labels"):
            self.raw_images = np.asarray(train_dataset.imgs)[self.indices]
            self.raw_labels = np.asarray(train_dataset.labels)[self.indices]

        # 兼容 torchvision CIFAR-10: data / targets
        elif hasattr(train_dataset, "data") and hasattr(train_dataset, "targets"):
            self.raw_images = np.asarray(train_dataset.data)[self.indices]
            self.raw_labels = np.asarray(train_dataset.targets)[self.indices].reshape(-1, 1)

        else:
            raise AttributeError(
                "train_dataset must have either MedMNIST-style "
                "`imgs` and `labels`, or torchvision-style `data` and `targets`."
            )

        self.train_transform = self._build_transform(
            train_dataset,
            augmentation_config,
            deterministic=False
        )
        self.eval_transform = self._build_transform(
            train_dataset,
            None,
            deterministic=True
        )

        self.client_dataset = _ClientDataset(
            self.raw_images,
            self.raw_labels,
            self.train_transform
        )
        self.eval_dataset = _ClientDataset(
            self.raw_images,
            self.raw_labels,
            self.eval_transform
        )

        pin_memory = "cuda" in str(device)

        self.train_loader = DataLoader(
            self.client_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin_memory
        )

        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory
        )

        self.model = model_fn().to(self.device)
        self.client_stats = self._build_client_stats()
    def _build_transform(self, train_dataset, augmentation_config, deterministic: bool = False):
        """
        构建客户端 transform。

        关键点：
        1. 保留原始 dataset.transform 中的 Resize / ToTensor / Normalize；
        2. 训练时将 augmentation 插入到 ToTensor 之前；
        3. eval 时不加随机增强；
        4. 兼容 MedMNIST 和 CIFAR-10。
        """
        base_transform = getattr(train_dataset, "transform", None)

        # 如果原始 transform 是 Compose，就拆开复用
        if isinstance(base_transform, transforms.Compose):
            base_tfms = list(base_transform.transforms)
        elif base_transform is not None:
            base_tfms = [base_transform]
        else:
            base_tfms = []

        pre_tensor_tfms = []
        post_tensor_tfms = []
        has_to_tensor = False
        has_normalize = False

        for t in base_tfms:
            if isinstance(t, transforms.ToTensor):
                has_to_tensor = True
                post_tensor_tfms.append(t)
            elif isinstance(t, transforms.Normalize):
                has_normalize = True
                post_tensor_tfms.append(t)
            else:
                # 例如 Resize、CenterCrop 等，应该保留在 ToTensor 前
                pre_tensor_tfms.append(t)

        aug_tfms = []
        if (not deterministic) and augmentation_config and augmentation_config.get("enabled", False):
            if augmentation_config.get("random_horizontal_flip", 0) > 0:
                aug_tfms.append(
                    transforms.RandomHorizontalFlip(
                        p=augmentation_config["random_horizontal_flip"]
                    )
                )

            if augmentation_config.get("random_rotation", 0) > 0:
                aug_tfms.append(
                    transforms.RandomRotation(
                        degrees=augmentation_config["random_rotation"]
                    )
                )

            if augmentation_config.get("color_jitter", 0) > 0:
                aug_tfms.append(
                    transforms.ColorJitter(
                        brightness=augmentation_config["color_jitter"]
                    )
                )

            if augmentation_config.get("random_affine", 0) > 0:
                t = augmentation_config["random_affine"]
                aug_tfms.append(
                    transforms.RandomAffine(
                        degrees=0,
                        translate=(t, t)
                    )
                )

        # 如果原始 transform 没有 ToTensor，则补一个
        if not has_to_tensor:
            post_tensor_tfms.insert(0, transforms.ToTensor())

        # 如果原始 transform 没有 Normalize，则根据通道数补默认 Normalize
        if not has_normalize:
            sample = self.raw_images[0]

            if sample.ndim == 2:
                channels = 1
            elif sample.ndim == 3:
                channels = sample.shape[2]
            else:
                raise ValueError(f"Unexpected sample shape for normalization: {sample.shape}")

            norm_mean = [0.5] * channels
            norm_std = [0.5] * channels
            post_tensor_tfms.append(
                transforms.Normalize(mean=norm_mean, std=norm_std)
            )

        transform_list = pre_tensor_tfms + aug_tfms + post_tensor_tfms
        return transforms.Compose(transform_list)
    def _build_client_stats(self):
        local_labels = self.raw_labels.flatten()
        num_samples = int(len(local_labels))
        unique, counts = np.unique(local_labels, return_counts=True)
        probs = counts / max(counts.sum(), 1)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum()) if len(probs) > 0 else 0.0
        min_class_count = int(counts.min()) if len(counts) > 0 else 0
        return {
            "num_samples": num_samples,
            "class_coverage": int(len(unique)),
            "label_entropy": entropy,
            "min_class_count": min_class_count,
        }

    def local_train(
        self,
        initial_state,
        epochs: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        prox_mu: float = 0.01,
        proximal_enabled: bool = True,
        optimizer_type: str = "adamw",
        momentum: float = 0.9,
        label_smoothing: float = 0.0,
        train_encoder: bool = True,
        train_head_only: bool = False,
        freeze_backbone_bn: bool = True,
    ):
        self.model.load_state_dict(initial_state, strict=True)
        self.model.train()
        if freeze_backbone_bn:
            self._freeze_backbone_batchnorm()

        initial_params = {name: param.detach().to(self.device).clone() for name, param in self.model.named_parameters()}

        params = self._select_trainable_params(train_encoder=train_encoder, train_head_only=train_head_only)
        if optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        for _ in range(epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if y.dim() > 1 and y.shape[1] == 1:
                    y = y.squeeze(1).long()
                optimizer.zero_grad()
                logits = self.model(x)
                ce_loss = criterion(logits, y)
                if proximal_enabled and prox_mu > 0:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if param.dtype.is_floating_point:
                            prox_loss = prox_loss + torch.norm(param - initial_params[name]) ** 2
                    loss = ce_loss + (prox_mu / 2.0) * prox_loss
                else:
                    loss = ce_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
            scheduler.step()

        cpu_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        meta = {
            "cid": int(self.cid),
            "num_samples": int(self.client_stats["num_samples"]),
            "class_coverage": int(self.client_stats["class_coverage"]),
            "label_entropy": float(self.client_stats["label_entropy"]),
            "min_class_count": int(self.client_stats["min_class_count"]),
        }
        return cpu_state, meta

    def _freeze_backbone_batchnorm(self):
        for module in self.model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False

    def _select_trainable_params(self, train_encoder: bool = True, train_head_only: bool = False):
        for p in self.model.parameters():
            p.requires_grad = False
        if train_head_only:
            params = list(self.model.head_parameters())
            for p in params:
                p.requires_grad = True
            return params
        params = []
        if train_encoder:
            enc = list(self.model.encoder_parameters())
            for p in enc:
                p.requires_grad = True
            params.extend(enc)
        head = list(self.model.head_parameters())
        for p in head:
            p.requires_grad = True
        params.extend(head)
        return params

    @torch.no_grad()
    def extract_classwise_statistics(
        self,
        model_state,
        object_level="O4",
        max_samples=None,
        num_classes=None,
    ):
        """
        返回按全局类别对齐的统计对象：
        counts:        [C]
        sum_features:  [C, D]
        diag_second:   [C, D]
        global_second: [D, D]
        class_second:  [C, D, D]   (仅 O5 需要)
        """
        self.model.load_state_dict(model_state, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # 推断全局类别数
        if num_classes is None:
            if hasattr(self.client_dataset, "labels"):
                num_classes = int(np.max(self.client_dataset.labels)) + 1
            else:
                num_classes = int(np.max(self.raw_labels)) + 1

        # 统计提取必须不用训练增强；建议你已有 deterministic eval loader
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if "cuda" in str(self.device) else False,
        )

        feat_dim = None
        counts = None
        sum_features = None
        diag_second = None
        global_second = None
        class_second = None

        seen = 0
        for x, y in eval_loader:
            x = x.to(self.device, non_blocking=True)
            if isinstance(y, torch.Tensor):
                y = y.view(-1).long().to(self.device, non_blocking=True)
            else:
                y = torch.as_tensor(y, dtype=torch.long, device=self.device)

            # feats = self.model.forward_features(x)   # [B, D]
            # feats = feats.detach()
            feats = self.model.forward_features(x)   # [B, D]
            # feats = F.normalize(feats, p=2, dim=1)   # 强制 L2 归一化
            feats = feats.detach()

            if feat_dim is None:
                feat_dim = feats.shape[1]
                counts = np.zeros((num_classes,), dtype=np.float64)
                sum_features = np.zeros((num_classes, feat_dim), dtype=np.float64)
                diag_second = np.zeros((num_classes, feat_dim), dtype=np.float64)
                global_second = np.zeros((feat_dim, feat_dim), dtype=np.float64)
                if object_level == "O5":
                    class_second = np.zeros((num_classes, feat_dim, feat_dim), dtype=np.float64)

            feats_np = feats.cpu().numpy()
            y_np = y.cpu().numpy()

            if max_samples is not None:
                remain = max_samples - seen
                if remain <= 0:
                    break
                if len(y_np) > remain:
                    feats_np = feats_np[:remain]
                    y_np = y_np[:remain]

            # 全局二阶矩
            global_second += feats_np.T @ feats_np

            # 按“全局类索引”填充
            for c in np.unique(y_np):
                mask = (y_np == c)
                fc = feats_np[mask]                  # [Nc, D]
                counts[c] += fc.shape[0]
                sum_features[c] += fc.sum(axis=0)
                diag_second[c] += (fc ** 2).sum(axis=0)

                if object_level == "O5":
                    class_second[c] += fc.T @ fc

            seen += len(y_np)

        payload = {
            "cid": int(self.cid),
            "num_classes": int(num_classes),
            "counts": counts,
            "sum_features": sum_features,
            "diag_second": diag_second,
            "global_second": global_second,
            "feature_dim": int(feat_dim),
            "object_level": object_level,
            "num_samples_used": int(counts.sum()),
            "class_coverage": int((counts > 0).sum()),
            "label_entropy": float(self.client_stats["label_entropy"]),
        }

        if object_level == "O5":
            payload["class_second"] = class_second

        return payload

class _ClientDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if img.ndim == 2:
            img = Image.fromarray(img, mode="L")
        elif img.ndim == 3 and img.shape[2] == 3:
            img = Image.fromarray(img)
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        label = self.labels[idx]
        if isinstance(label, np.ndarray):
            label = int(label.item()) if label.size == 1 else label
        else:
            label = int(label)

        if self.transform is not None:
            img = self.transform(img)

        return img, label