import numpy as np


def _normalize_labels(labels):
    """
    将不同数据集的标签统一为一维 int numpy array。

    支持：
    - MedMNIST: shape 可能是 (N, 1)
    - CIFAR-10: list[int]
    - 普通 numpy array / list
    """
    labels = np.asarray(labels).reshape(-1).astype(int)
    if labels.size == 0:
        raise ValueError("labels is empty.")
    return labels


def dirichlet_label_partition(
    labels,
    num_clients,
    alpha,
    seed=42,
    return_stats=True,
    verbose=False
):
    """
    标签偏斜划分。

    每个类别的样本按照 Dirichlet(alpha) 分配到不同客户端。
    alpha 越小，label skew 越强。

    Args:
        labels: 一维标签数组，或可被压平的一维/二维标签。
        num_clients: 客户端数量。
        alpha: Dirichlet concentration parameter。
        seed: 随机种子。
        return_stats: 是否返回划分统计信息。
        verbose: 是否打印划分摘要。

    Returns:
        若 return_stats=True:
            client_indices, stats
        否则:
            client_indices
    """
    rng = np.random.default_rng(seed)
    labels = _normalize_labels(labels)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    num_classes = int(labels.max() + 1)

    idx_by_class = [np.where(labels == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    client_class_counts = np.zeros((num_clients, num_classes), dtype=int)

    for c in range(num_classes):
        class_idx = idx_by_class[c]
        if len(class_idx) == 0:
            continue

        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        proportions = np.maximum(proportions, 1e-8)
        proportions = proportions / proportions.sum()

        splits = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]
        shards = np.split(class_idx, splits)

        for i in range(num_clients):
            if i < len(shards) and len(shards[i]) > 0:
                client_indices[i].extend(shards[i].tolist())
                client_class_counts[i, c] = len(shards[i])

    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i], dtype=int)

    stats = (
        _compute_partition_stats(client_indices, labels, client_class_counts)
        if return_stats else None
    )

    if verbose and stats:
        _print_partition_summary(stats, alpha, "label")

    return (client_indices, stats) if return_stats else client_indices


def quantity_skew_partition(
    labels,
    num_clients,
    alpha_q,
    seed=42,
    return_stats=True,
    verbose=False
):
    """
    数量偏斜划分。

    客户端样本量服从 Dirichlet(alpha_q) 分布。
    alpha_q 越小，客户端样本量差异越大。

    注意：
    该函数主要改变每个客户端的数据量，而不是刻意制造标签偏斜。
    """
    rng = np.random.default_rng(seed)
    labels = _normalize_labels(labels)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha_q <= 0:
        raise ValueError("alpha_q must be positive.")

    total = len(labels)
    num_classes = int(labels.max() + 1)

    props = rng.dirichlet(alpha_q * np.ones(num_clients))
    sizes = (props * total).astype(int)

    # 确保总样本数严格等于 total
    sizes[-1] = total - sizes[:-1].sum()

    indices = np.arange(total)
    rng.shuffle(indices)

    client_indices = []
    start = 0
    for size in sizes:
        end = start + int(size)
        client_indices.append(indices[start:end].astype(int))
        start = end

    client_class_counts = np.zeros((num_clients, num_classes), dtype=int)

    for i, idx in enumerate(client_indices):
        if len(idx) == 0:
            continue
        unique, counts = np.unique(labels[idx], return_counts=True)
        client_class_counts[i, unique] = counts

    stats = (
        _compute_partition_stats(client_indices, labels, client_class_counts)
        if return_stats else None
    )

    if verbose and stats:
        _print_partition_summary(stats, alpha_q, "quantity")

    return (client_indices, stats) if return_stats else client_indices


def mixed_partition(
    labels,
    num_clients,
    alpha_label,
    alpha_q,
    seed=42,
    return_stats=True,
    verbose=False
):
    """
    混合异质性划分：同时引入标签偏斜和数量偏斜。

    当前实现思路：
    1. 先根据 Dirichlet(alpha_q) 生成每个客户端目标样本量；
    2. 再根据每个客户端的标签偏好分配样本；
    3. 保证每个样本最多分配一次，不做重复采样。

    注意：
    该函数适合作为扩展实验或演示。
    当前正式论文主线建议优先使用：
    - dirichlet_label_partition
    - quantity_skew_partition
    """
    rng = np.random.default_rng(seed)
    labels = _normalize_labels(labels)

    if num_clients <= 0:
        raise ValueError("num_clients must be positive.")
    if alpha_label <= 0:
        raise ValueError("alpha_label must be positive.")
    if alpha_q <= 0:
        raise ValueError("alpha_q must be positive.")

    total = len(labels)
    num_classes = int(labels.max() + 1)

    # 1. 生成客户端目标样本量
    client_props = rng.dirichlet(alpha_q * np.ones(num_clients))
    target_sizes = (client_props * total).astype(int)
    target_sizes[-1] = total - target_sizes[:-1].sum()

    remaining_capacity = target_sizes.copy()

    # 2. 每个客户端生成标签偏好
    client_label_pref = rng.dirichlet(
        alpha_label * np.ones(num_classes),
        size=num_clients
    )

    idx_by_class = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(idx_by_class[c])

    client_indices = [[] for _ in range(num_clients)]
    client_class_counts = np.zeros((num_clients, num_classes), dtype=int)

    # 3. 逐类别分配样本
    for c in range(num_classes):
        class_indices = idx_by_class[c]
        if len(class_indices) == 0:
            continue

        # 按客户端对该类别的偏好与剩余容量共同决定分配权重
        weights = client_label_pref[:, c] * np.maximum(remaining_capacity, 0)

        if weights.sum() <= 0:
            # 如果所有容量都满了，停止分配
            break

        weights = weights / weights.sum()

        for sample_idx in class_indices:
            valid_clients = np.where(remaining_capacity > 0)[0]

            if len(valid_clients) == 0:
                break

            valid_weights = weights[valid_clients]
            if valid_weights.sum() <= 0:
                valid_weights = np.ones(len(valid_clients)) / len(valid_clients)
            else:
                valid_weights = valid_weights / valid_weights.sum()

            chosen_client = rng.choice(valid_clients, p=valid_weights)

            client_indices[chosen_client].append(sample_idx)
            client_class_counts[chosen_client, c] += 1
            remaining_capacity[chosen_client] -= 1

    for i in range(num_clients):
        rng.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i], dtype=int)

    stats = (
        _compute_partition_stats(client_indices, labels, client_class_counts)
        if return_stats else None
    )

    if verbose and stats:
        _print_partition_summary(stats, (alpha_label, alpha_q), "mixed")

    return (client_indices, stats) if return_stats else client_indices


def _compute_partition_stats(client_indices, labels, client_class_counts):
    """
    计算划分统计信息。
    """
    labels = _normalize_labels(labels)
    samples_per_client = [len(idx) for idx in client_indices]
    nonzero_sizes = [s for s in samples_per_client if s > 0]

    stats = {
        "num_clients": len(client_indices),
        "total_samples": int(sum(samples_per_client)),
        "original_samples": int(len(labels)),
        "samples_per_client": [int(s) for s in samples_per_client],
        "client_class_counts": client_class_counts.astype(int).tolist(),
        "class_coverage_per_client": np.sum(client_class_counts > 0, axis=1).astype(int).tolist(),
        "client_coverage_per_class": np.sum(client_class_counts > 0, axis=0).astype(int).tolist(),
        "imbalance_ratio": None,
        "num_empty_clients": int(sum(s == 0 for s in samples_per_client)),
        "entropy_per_client": []
    }

    if nonzero_sizes:
        stats["imbalance_ratio"] = float(max(nonzero_sizes) / min(nonzero_sizes))

    for i in range(len(client_indices)):
        row = client_class_counts[i]
        if row.sum() > 0:
            probs = row / row.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            stats["entropy_per_client"].append(float(entropy))

    return stats


def _print_partition_summary(stats, alpha, skew_type):
    """
    打印划分摘要。
    """
    print(f"\n{'=' * 50}")
    print(f"📊 数据划分摘要 (type={skew_type}, alpha={alpha})")
    print(f"{'=' * 50}")
    print(f"客户端数: {stats['num_clients']}")
    print(f"总样本: {stats['total_samples']} / {stats['original_samples']}")
    print(f"空客户端数: {stats.get('num_empty_clients', 0)}")

    if stats["original_samples"] > 0:
        print(f"样本恢复率: {stats['total_samples'] / stats['original_samples']:.2%}")

    if stats["samples_per_client"]:
        print(
            f"\n📈 样本量分布: "
            f"mean={np.mean(stats['samples_per_client']):.1f}, "
            f"min={min(stats['samples_per_client'])}, "
            f"max={max(stats['samples_per_client'])}, "
            f"imbalance={stats['imbalance_ratio'] if stats['imbalance_ratio'] is not None else 'NA'}"
        )

    print(
        f"\n🎯 类别覆盖: "
        f"平均每客户端覆盖 {np.mean(stats['class_coverage_per_client']):.1f} 类"
    )

    if stats["entropy_per_client"]:
        print(
            f"🔢 标签熵: "
            f"mean={np.mean(stats['entropy_per_client']):.3f}, "
            f"min={min(stats['entropy_per_client']):.3f}, "
            f"max={max(stats['entropy_per_client']):.3f}"
        )

    print(f"{'=' * 50}\n")