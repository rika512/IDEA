# IDEA Reproduction on MSVR310

## 1. Overview

This repository is based on the official implementation of:

**IDEA: Inverted Text with Cooperative Deformable Aggregation for Multi-modal Object Re-Identification (CVPR 2025)**

I reproduced IDEA on the **MSVR310** dataset to become familiar with the complete research workflow, including:

- Reading and understanding a research codebase
- Preparing datasets and pretrained weights
- Configuring a GPU training environment
- Running training and evaluation
- Saving and loading checkpoints
- Debugging reproduction issues
- Managing reproduction code with Git and GitHub

This repository is a **reproduction and learning record**, not a new implementation of IDEA.

---

## 2. Environment

The experiment was run on AutoDL with the following environment:

- GPU: NVIDIA GeForce RTX 4090 D
- Python: 3.10.8
- PyTorch: 2.1.2+cu118
- TorchVision: 0.16.2+cu118
- CUDA: 11.8

The main Python dependencies were installed from:

```bash
python -m pip install -r requirements.txt
```

---

## 3. Dataset

Dataset:

**MSVR310**

The dataset was stored locally with the following structure:

```text
DATA/
└── MSVR310/
    ├── bounding_box_train/
    ├── bounding_box_test/
    ├── query/
    ├── query3/
    ├── test/
    ├── train/
    └── text/
```

The dataset files are not included in this repository.

The dataset root in:

```text
configs/MSVR310/IDEA.yml
```

was changed from:

```yaml
ROOT_DIR: '..'
```

to:

```yaml
ROOT_DIR: './DATA'
```

so that the project can locate:

```text
./DATA/MSVR310/
```

when executed from the repository root.

---

## 4. CLIP Pretrained Weight

The pretrained CLIP weight used in this reproduction was:

```text
PTH/ViT-B-16.pt
```

The original code contained an absolute path from the authors' environment.

In:

```text
modeling/clip/make_model_clipreid.py
```

the path was changed to:

```python
model_path = './PTH/ViT-B-16.pt'
```

The pretrained weight itself is not included in this repository.

---

## 5. Training Configuration

The main MSVR310 configuration was:

```text
configs/MSVR310/IDEA.yml
```

Important settings:

| Setting | Value |
|---|---:|
| Backbone | ViT-B-16 |
| Epochs | 50 |
| Train batch size | 64 |
| Test batch size | 128 |
| Instances per identity | 8 |
| Optimizer | Adam |
| Base learning rate | 0.00035 |
| Seed | 1111 |
| Image size | 128 × 256 |
| Evaluation period | 1 epoch |

The IDEA modules used in the configuration included:

```yaml
INVERSE: True
DA: True
DA_SHARE: True
DIRECT: 1
TEXT_PROMPT: 1
OFF_FAC: 25.0
```

---

## 6. Training

Training was started from the repository root using:

```bash
python train.py --config_file ./configs/MSVR310/IDEA.yml
```

The model was trained for the complete **50 epochs**.

The training process produced two checkpoints:

```text
IDEAbest.pth
IDEA_50.pth
```

- `IDEAbest.pth`: checkpoint corresponding to the best validation result
- `IDEA_50.pth`: checkpoint saved at epoch 50

The checkpoint files are stored locally and are not committed to GitHub because of their large size.

---

## 7. Test Script Fix

The original `test.py` contained:

```python
model.load_param("~")
```

This caused the test script to attempt to load `"~"` as the checkpoint path.

It was changed to:

```python
model.load_param(cfg.TEST.WEIGHT)
```

so that the checkpoint path supplied through the configuration or command line is used correctly.

The best checkpoint was evaluated using:

```bash
python test.py \
    --config_file ./configs/MSVR310/IDEA.yml \
    TEST.WEIGHT ./IDEA_MSVR310/IDEAbest.pth
```

---

## 8. Output Files

The reproduction produced:

```text
IDEA_50.pth
IDEAbest.pth
train_log.txt
test_log.txt
```

The two log files record the complete training and evaluation processes.

Large datasets, pretrained weights, and model checkpoints are kept locally instead of being committed to the Git repository.

---

## 9. Notes

This reproduction was mainly used to learn and complete the full workflow from a research paper to an executable experiment:

```text
Paper
  ↓
Code reading
  ↓
Dataset preparation
  ↓
Environment setup
  ↓
Training
  ↓
Checkpoint
  ↓
Evaluation
  ↓
Result analysis
```

The reproduction result is recorded separately in `RESULTS.md`.