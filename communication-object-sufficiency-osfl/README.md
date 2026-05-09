# SOS-HAMOC: Strict One-Shot Communication-Object Reconstruction

**Code for:** *Non-Monotonic Communication-Object Sufficiency in Strict One-Shot Federated Classification*

This repository contains the experimental code for studying communication-object sufficiency in strict one-shot federated classification. The code supports parameter-upload baselines and structured statistical communication objects, including diagonal Gaussian, shared-covariance LDA, and class-wise covariance QDA reconstruction.

The main idea is to ask a practical one-shot federated learning question: **what should a client communicate when it can upload only once?** Rather than treating one-shot FL only as a parameter-fusion problem, this code evaluates whether different client-side statistical objects are sufficient for reconstructing a reliable classifier at the server.

---

## 1. Main features

- Strict one-shot federated classification protocol.
- Label-skewed non-IID partitioning with Dirichlet sampling.
- Full-parameter one-shot baseline (`Param-BaseOSFL`).
- Statistical communication-object reconstruction:
  - O2 / nearest-class-mean prototype object.
  - O3 / diagonal Gaussian object.
  - O4 / shared-covariance LDA object.
  - O5 / class-wise covariance QDA object.
- ResNet18-based image encoder with ImageNet initialization.
- CIFAR-100 high-class-count stress-test configurations.
- Seed-level evaluation and result logging.

---

## 2. Repository structure

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

Before public release, remove generated files such as `__pycache__/`, local `outputs/`, temporary logs, and any machine-specific paths.

---

## 3. Environment

The code was developed and tested with Python 3.10/3.11 and PyTorch. A CUDA-enabled GPU is recommended for CIFAR-100 experiments.

Example environment setup:

```bash
conda create -n sos-hamoc python=3.10 -y
conda activate sos-hamoc
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn pyyaml pillow medmnist
```

For reproducible release, it is recommended to provide an `environment.yml` or `requirements.txt` generated from the final environment.

---

## 4. Datasets

This code uses public benchmark datasets.

- CIFAR-10 and CIFAR-100 are loaded through `torchvision.datasets`.
- MedMNIST datasets are supported through the `medmnist` package.

By default, datasets are stored under:

```text
./data
```

The repository should not include downloaded raw datasets. They will be downloaded or prepared locally according to the dataset loader.

---

## 5. Running experiments

Each experiment is controlled by a YAML configuration file. The general command is:

```bash
python run_main.py configs/<config_name>.yaml
```

### CIFAR-100 parameter-upload baseline

```bash
python run_main.py configs/cifar100_labelskew_parameter_baseosfl.yaml
```

### CIFAR-100 O3 / diagonal Gaussian

```bash
python run_main.py configs/cifar100_labelskew_msco_o3_diag_gaussian.yaml
```

### CIFAR-100 O4 / shared-covariance LDA

```bash
python run_main.py configs/cifar100_labelskew_msco_o4_lda.yaml
```

### CIFAR-100 O5 / class-wise covariance QDA

```bash
python run_main.py configs/cifar100_labelskew_msco_o5_qda.yaml
```

To specify a GPU manually, edit the `hardware.gpu_id` field in the YAML file or use `CUDA_VISIBLE_DEVICES`, for example:

```bash
CUDA_VISIBLE_DEVICES=0 python run_main.py configs/cifar100_labelskew_msco_o4_lda.yaml
CUDA_VISIBLE_DEVICES=1 python run_main.py configs/cifar100_labelskew_msco_o5_qda.yaml
```

---

## 6. Outputs

Experiment outputs are saved under the directory specified by `output_dir`, usually:

```text
outputs/
```

Typical outputs include:

- `results.csv`: appended seed-level results.
- `summary.json`: mean and standard deviation across seeds.
- per-run folders containing model states, reconstructed statistical heads, global statistics, and diagnostic summaries.

Large checkpoint files such as `global_model.pt` should usually not be committed to GitHub. If needed, release them separately through Zenodo, Figshare, or another archival repository.

---

## 7. Reproducing the CIFAR-100 high-class-count stress test

The CIFAR-100 stress test in the manuscript compares:

- `Param-BaseOSFL`
- `O3 / Diag-Gaussian`
- `O4 / LDA`
- `O5 / QDA`

under a strict one-shot label-skewed setting with:

```text
number of clients = 20
Dirichlet alpha = 0.1
seeds = 42, 52, 62
feature dimension = 512
```

The corresponding configurations are provided in `configs/`.

---

## 8. Repository hygiene

The repository excludes raw datasets, generated outputs, model checkpoints, temporary logs, and machine-specific paths. Large artifacts should be released separately through an archival repository if needed.

---

## 9. Citation

If you use this code, please cite the corresponding manuscript:

```bibtex
@article{li2026nonmonotonic,
  title   = {Non-Monotonic Communication-Object Sufficiency in Strict One-Shot Federated Classification},
  author  = {Li, Ning},
  journal = {Under review},
  year    = {2026}
}
```

---

## 10. Contact

For questions about the code or experiments, please contact:

**Ning Li**  
Department of Biomedical Engineering, Changzhi Medical College  
Email: cfcfcfpl@163.com

---

## 11.License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

