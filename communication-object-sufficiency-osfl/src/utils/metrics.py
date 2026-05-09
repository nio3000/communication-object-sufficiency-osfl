import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

@torch.no_grad()
def evaluate(model, dataset, device=None, batch_size=256, num_classes=None):
    """
    评估模型在给定数据集上的多维度性能
    返回:
        dict: {
            'accuracy': float,
            'macro_f1': float,
            'balanced_acc': float,
            'auc': float (或 NaN 如果不可计算)
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds, all_labels, all_probs = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        # 处理标签格式
        if isinstance(y, torch.Tensor):
            y = y.squeeze().long().to(device, non_blocking=True)
        else:
            y = torch.as_tensor(y, dtype=torch.long, device=device)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        all_probs.append(probs.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.cat(all_probs).numpy()

    # 基础指标
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(labels, preds)

    # AUC (one-vs-rest)
    if num_classes is None:
        num_classes = probs.shape[1]
    try:
        if num_classes == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "auc": auc
    }