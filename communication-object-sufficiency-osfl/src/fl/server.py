from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

from src.utils.metrics import evaluate
import torch.nn.functional as F

class FLServer:
    def __init__(self, model_fn, device):
        self.device = device
        self.model_fn = model_fn
        self.global_model = model_fn().to(device)
        self.external_head = None

    def get_state(self):
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def set_state(self, state):
        new_state = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in state.items()}
        self.global_model.load_state_dict(new_state, strict=True)

    def set_external_head(self, head):
        self.external_head = head

    def clear_external_head(self):
        self.external_head = None

    @torch.no_grad()
    def test(self, dataset, batch_size=256, num_classes=None):
        self.global_model.eval()
        return evaluate(self.global_model, dataset, device=self.device, batch_size=batch_size, num_classes=num_classes)

    @torch.no_grad()
    def test_with_external_head(self, dataset, batch_size=256, num_classes=None):
        if self.external_head is None:
            raise RuntimeError("external_head is not set")
        self.global_model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        all_preds, all_labels, all_probs = [], [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            if isinstance(y, torch.Tensor):
                y = y.squeeze().long().to(self.device, non_blocking=True)
            else:
                y = torch.as_tensor(y, dtype=torch.long, device=self.device)
            # feats = self.global_model.forward_features(x)
            # logits = self.external_head.predict_logits(feats)
            feats = self.global_model.forward_features(x)
            # feats = F.normalize(feats, p=2, dim=1)   # 必须与 Client 保持严格对齐
            logits = self.external_head.predict_logits(feats)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_probs.append(probs.cpu())

        preds = torch.cat(all_preds).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = torch.cat(all_probs).numpy()
        if num_classes is None:
            num_classes = probs.shape[1]
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
        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "balanced_acc": balanced_acc,
            "auc": auc,
        }

    def save_model(self, path):
        torch.save(self.global_model.state_dict(), path)
        print(f"✅ 模型已成功保存至: {path}")
