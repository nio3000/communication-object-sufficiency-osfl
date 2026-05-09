import os
import numpy as np

from medmnist import PathMNIST, DermaMNIST, OrganAMNIST
from torchvision import datasets, transforms
from torch.utils.data import Dataset

def get_transform(dataset_name: str, train: bool = False):
    """
    根据数据集返回适当的预处理与归一化参数。

    注意：
    1. 这里默认不放强数据增强，避免和你现有 pipeline 里的 data_augmentation 重复。
    2. CIFAR-10 使用 Resize(224, 224)，更适配 torchvision 标准 ResNet18 / ImageNet-pretrained encoder。
    """
    name = dataset_name.lower()

    if name == "pathmnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    if name == "dermamnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    if name == "organamnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5],
                                 std=[0.5])
        ])

    if name == "cifar10":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            )
        ])
    if name == "cifar100":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        ])    
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _check_npz_exists(data_root: str, filename: str):
    npz_path = os.path.join(data_root, filename)
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"找不到数据文件: {npz_path}。请确认 {filename} 已放在 data 文件夹下。"
        )
    return npz_path


def get_targets(dataset):
    """
    统一获取不同数据集的标签。

    支持：
    - MedMNIST: dataset.labels
    - CIFAR-10: dataset.targets
    - TargetSubset: dataset.targets
    - torch Subset-like dataset: dataset.dataset + dataset.indices
    """
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        return np.asarray(targets).reshape(-1).astype(int).tolist()

    if hasattr(dataset, "labels"):
        labels = dataset.labels
        return np.asarray(labels).reshape(-1).astype(int).tolist()

    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base_targets = get_targets(dataset.dataset)
        return [int(base_targets[i]) for i in dataset.indices]

    raise AttributeError(
        "Cannot find labels. Dataset must have `targets`, `labels`, "
        "or be a subset with `dataset` and `indices`."
    )


class TargetSubset(Dataset):
    """
    保留 imgs / labels / data / targets 字段的子集包装器。

    目的：
    1. 兼容旧版 FLClient 对 MedMNIST 风格字段 imgs / labels 的要求；
    2. 兼容 CIFAR-10 原始字段 data / targets；
    3. 保留 transform，使 __getitem__ 仍然可以正常 Resize / ToTensor / Normalize。
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(map(int, indices))

        self.transform = getattr(dataset, "transform", None)
        self.target_transform = getattr(dataset, "target_transform", None)

        # ---------- labels / targets ----------
        base_targets = get_targets(dataset)
        self.targets = [int(base_targets[i]) for i in self.indices]
        self.labels = np.asarray(self.targets).reshape(-1, 1)

        # ---------- imgs / data ----------
        if hasattr(dataset, "imgs"):
            base_imgs = np.asarray(dataset.imgs)
            self.imgs = base_imgs[self.indices]
        elif hasattr(dataset, "data"):
            base_imgs = np.asarray(dataset.data)
            self.imgs = base_imgs[self.indices]
        else:
            raise AttributeError(
                "Base dataset must have either `imgs` or `data` attribute."
            )

        # CIFAR-10 风格字段也保留一份
        self.data = self.imgs

        # 尽量保留 info；CIFAR-10 没有 info 也没关系
        if hasattr(dataset, "info"):
            self.info = dataset.info

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    
def split_train_val(dataset, val_ratio: float = 0.1, seed: int = 42):
    """
    将 CIFAR-10 原始 train 集划分为 train / val。

    CIFAR-10 官方只有 train/test，没有 val。
    为了兼容你现有的 train, val, test 三返回值结构，
    这里从原始 train 中固定划分 10% 作为 val。
    """
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_set = TargetSubset(dataset, train_indices)
    val_set = TargetSubset(dataset, val_indices)

    return train_set, val_set


def load_pathmnist(data_root: str):
    tfm = get_transform("pathmnist")
    _check_npz_exists(data_root, "pathmnist.npz")

    train = PathMNIST(split="train", root=data_root, transform=tfm, download=False)
    val = PathMNIST(split="val", root=data_root, transform=tfm, download=False)
    test = PathMNIST(split="test", root=data_root, transform=tfm, download=False)

    return train, val, test


def load_dermamnist(data_root: str):
    tfm = get_transform("dermamnist")
    _check_npz_exists(data_root, "dermamnist.npz")

    train = DermaMNIST(split="train", root=data_root, transform=tfm, download=False)
    val = DermaMNIST(split="val", root=data_root, transform=tfm, download=False)
    test = DermaMNIST(split="test", root=data_root, transform=tfm, download=False)

    return train, val, test


def load_organamnist(data_root: str):
    tfm = get_transform("organamnist")
    _check_npz_exists(data_root, "organamnist.npz")

    train = OrganAMNIST(split="train", root=data_root, transform=tfm, download=False)
    val = OrganAMNIST(split="val", root=data_root, transform=tfm, download=False)
    test = OrganAMNIST(split="test", root=data_root, transform=tfm, download=False)

    return train, val, test


def load_cifar10(data_root: str):
    """
    加载 CIFAR-10。

    返回：
    - train: 从 CIFAR-10 train 中划出的 90%
    - val: 从 CIFAR-10 train 中划出的 10%
    - test: CIFAR-10 官方 test

    注意：
    如果本地没有 CIFAR-10，会自动下载到 data_root。
    """
    tfm = get_transform("cifar10")

    train_full = datasets.CIFAR10(
        root=data_root,
        train=True,
        transform=tfm,
        download=True
    )

    test = datasets.CIFAR10(
        root=data_root,
        train=False,
        transform=tfm,
        download=True
    )

    train, val = split_train_val(
        train_full,
        val_ratio=0.1,
        seed=42
    )

    return train, val, test

def load_cifar100(data_root: str):
    """
    加载 CIFAR-100。

    返回：
    - train: 从 CIFAR-100 train 中划出的 90%
    - val: 从 CIFAR-100 train 中划出的 10%
    - test: CIFAR-100 官方 test

    注意：
    如果本地没有 CIFAR-100，会自动下载到 data_root。
    """
    tfm = get_transform("cifar100")

    train_full = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=tfm,
        download=True
    )

    test = datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=tfm,
        download=True
    )

    train, val = split_train_val(
        train_full,
        val_ratio=0.1,
        seed=42
    )

    return train, val, test

DATASET_LOADERS = {
    "pathmnist": load_pathmnist,
    "dermamnist": load_dermamnist,
    "organamnist": load_organamnist,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
}