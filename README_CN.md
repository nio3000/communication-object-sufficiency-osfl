[![DOI](https://zenodo.org/badge/1233577082.svg)](https://doi.org/10.5281/zenodo.20093074)

# 中文说明

# SOS-HAMOC：严格单轮联邦分类中的通信对象重构代码

**对应论文：**《Non-Monotonic Communication-Object Sufficiency in Strict One-Shot Federated Classification》

本仓库包含用于研究严格单轮联邦分类中“通信对象充分性”的实验代码。代码支持完整参数上传基线，以及基于统计通信对象的服务器端分类器重构，包括对角高斯、共享协方差 LDA 和类别级协方差 QDA 等闭式重构方式。

本文关注的问题是：**在客户端只能上传一次的 strict one-shot federated learning 场景中，客户端到底应该上传什么对象，服务器才能重构出可靠分类器？** 与其仅把单轮联邦学习看作参数融合问题，本代码用于评估不同客户端统计对象是否足以支持服务器端分类器重构。

---

## 1. 主要功能

- 严格单轮联邦分类协议。
- 基于 Dirichlet 分布的 label-skew 非独立同分布划分。
- 完整参数上传基线（`Param-BaseOSFL`）。
- 统计通信对象重构：
  - O2 / 最近类均值原型对象；
  - O3 / 对角高斯对象；
  - O4 / 共享协方差 LDA 对象；
  - O5 / 类别级协方差 QDA 对象。
- 基于 ResNet18 的图像编码器，并使用 ImageNet 预训练初始化。
- CIFAR-100 高类别数压力测试配置。
- 三种子评估与结果记录。

---

## 2. 仓库结构

```text
SOS-HAMOC/
├── configs/
│   ├── pathmnist_*.yaml
│   ├── dermamnist_*.yaml
│   ├── organamnist_*.yaml
│   ├── cifar10_*.yaml
│   └── cifar100_*.yaml
├── src/
│   ├── data/
│   │   ├── datasets.py
│   │   └── partition.py
│   ├── fl/
│   │   ├── aggregation.py
│   │   ├── algorithms.py
│   │   ├── client.py
│   │   └── server.py
│   ├── models/
│   │   ├── cnn.py
│   │   └── resnet.py
│   └── utils/
│       ├── metrics.py
│       └── seed.py
├── run_main.py
└── README.md
```

公开发布前建议删除 `__pycache__/`、本地 `outputs/`、临时日志和任何机器相关路径。

---

## 3. 运行环境

代码主要在 Python 3.10/3.11 和 PyTorch 环境下运行。CIFAR-100 实验建议使用 CUDA GPU。

示例环境配置：

```bash
conda create -n sos-hamoc python=3.10 -y
conda activate sos-hamoc
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn pyyaml pillow medmnist
```

正式公开时，建议同时提供由最终运行环境导出的 `environment.yml` 或 `requirements.txt`。

---

## 4. 数据集

本代码使用公开 benchmark 数据集。

- CIFAR-10 和 CIFAR-100 通过 `torchvision.datasets` 加载。
- MedMNIST 数据集通过 `medmnist` 包支持。

默认数据目录为：

```text
./data
```

仓库中不应包含下载后的原始数据文件。数据可根据加载器在本地下载或准备。

---

## 5. 运行实验

每个实验由 YAML 配置文件控制。通用运行命令为：

```bash
python run_main.py configs/<config_name>.yaml
```

### CIFAR-100 完整参数上传基线

```bash
python run_main.py configs/cifar100_labelskew_parameter_baseosfl.yaml
```

### CIFAR-100 O3 / 对角高斯

```bash
python run_main.py configs/cifar100_labelskew_msco_o3_diag_gaussian.yaml
```

### CIFAR-100 O4 / 共享协方差 LDA

```bash
python run_main.py configs/cifar100_labelskew_msco_o4_lda.yaml
```

### CIFAR-100 O5 / 类别级协方差 QDA

```bash
python run_main.py configs/cifar100_labelskew_msco_o5_qda.yaml
```

如需指定 GPU，可以修改 YAML 中的 `hardware.gpu_id` 字段，也可以使用 `CUDA_VISIBLE_DEVICES`：

```bash
CUDA_VISIBLE_DEVICES=0 python run_main.py configs/cifar100_labelskew_msco_o4_lda.yaml
CUDA_VISIBLE_DEVICES=1 python run_main.py configs/cifar100_labelskew_msco_o5_qda.yaml
```

---

## 6. 输出文件

实验输出保存在配置文件中 `output_dir` 指定的目录下，通常为：

```text
outputs/
```

常见输出包括：

- `results.csv`：追加保存的种子级结果；
- `summary.json`：多种子的均值与标准差；
- 每次运行对应的子目录，其中包含模型状态、统计分类头、全局统计量和诊断结果。

`global_model.pt` 等大型模型文件通常不建议直接提交到 GitHub。如确需公开，可通过 Zenodo、Figshare 等仓库单独归档。

---

## 7. 复现 CIFAR-100 高类别数压力测试

论文中的 CIFAR-100 压力测试比较了：

- `Param-BaseOSFL`
- `O3 / Diag-Gaussian`
- `O4 / LDA`
- `O5 / QDA`

实验采用 strict one-shot label-skew 设置：

```text
客户端数量 = 20
Dirichlet alpha = 0.1
随机种子 = 42, 52, 62
特征维度 = 512
```

对应配置文件已经放在 `configs/` 目录下。

---

## 8. 仓库整理说明

本仓库不包含原始数据集、实验输出、模型权重、临时日志和机器相关路径。如需公开大型模型或结果文件，建议通过 Zenodo、Figshare 等归档平台单独发布。

---

## 9. 引用

如果使用本代码，请引用对应论文：

```bibtex
@article{li2026nonmonotonic,
  title   = {Non-Monotonic Communication-Object Sufficiency in Strict One-Shot Federated Classification},
  author  = {Li, Ning},
  journal = {Under review},
  year    = {2026}
}
```

---

## 10. 联系方式

如对代码或实验有疑问，请联系：

**Ning Li**  
Department of Biomedical Engineering, Changzhi Medical College  
Email: cfcfcfpl@163.com

---

## 11.许可证

本项目代码采用 MIT License 开源。具体内容见 [LICENSE](LICENSE) 文件。
