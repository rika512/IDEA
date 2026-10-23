# IDEA MSVR310 Reproduction Results

## 1. Experiment

Dataset:

**MSVR310**

Model:

**IDEA**

Backbone:

**ViT-B-16**

Training configuration:

| Setting | Value |
|---|---:|
| Epochs | 50 |
| Batch size | 64 |
| Test batch size | 128 |
| Seed | 1111 |
| Optimizer | Adam |
| Base learning rate | 0.00035 |
| GPU | RTX 4090 D |

---

## 2. Best Validation Result

The best validation result appeared at approximately **epoch 32**.

The metric used by the training code for selecting the best checkpoint was based on the `LOCAL_t` feature.

| Metric | Result |
|---|---:|
| mAP | **46.4%** |
| Rank-1 | **62.8%** |
| Rank-5 | **76.6%** |
| Rank-10 | **83.1%** |

The corresponding checkpoint was saved as:

```text
IDEAbest.pth
```

---

## 3. Final Epoch

Training successfully finished all **50 epochs**.

At epoch 50, one of the reported combined-feature evaluation results was:

| Metric | Result |
|---|---:|
| mAP | 46.0% |
| Rank-1 | 60.9% |
| Rank-5 | 75.6% |
| Rank-10 | 80.7% |

The epoch-50 checkpoint was saved as:

```text
IDEA_50.pth
```

The final epoch was therefore not the best-performing checkpoint, which is why the separately saved `IDEAbest.pth` was used for final evaluation.

---

## 4. Evaluation of the Best Checkpoint

The best checkpoint was evaluated independently using:

```bash
python test.py \
    --config_file ./configs/MSVR310/IDEA.yml \
    TEST.WEIGHT ./IDEA_MSVR310/IDEAbest.pth
```

Selected evaluation results are shown below.

| Feature / Search Pattern | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| T_RGB | 39.4% | 56.0% | 71.6% | 78.7% |
| T_NIR | 32.7% | 50.6% | 72.6% | 79.5% |
| T_TIR | 10.5% | 20.6% | 37.1% | 44.0% |
| T_RGB + T_NIR | 44.7% | 61.3% | 76.3% | 80.4% |
| T_RGB + T_TIR | 40.3% | 54.5% | 72.9% | 80.0% |
| T_NIR + T_TIR | 35.2% | 51.1% | 71.2% | 78.2% |
| T_RGB + T_NIR + T_TIR | 46.4% | 62.1% | 77.0% | 82.4% |
| LOCAL | 46.3% | 61.4% | 77.0% | 82.2% |
| LOCAL_v | 45.7% | 60.2% | 76.8% | 82.2% |
| LOCAL_t | **46.4%** | **62.8%** | **76.6%** | **83.1%** |

---

## 5. Reproduction Summary

The complete pipeline was successfully executed:

- Dataset loading
- CLIP pretrained weight loading
- 50-epoch training
- Validation after each epoch
- Best-checkpoint saving
- Final checkpoint saving
- Independent checkpoint loading
- Test-time evaluation

Final reproduced best result:

```text
mAP:     46.4%
Rank-1:  62.8%
Rank-5:  76.6%
Rank-10: 83.1%
```

The following experiment artifacts are kept locally:

```text
IDEAbest.pth
IDEA_50.pth
train_log.txt
test_log.txt
```

The `.pth` checkpoint files are not uploaded to GitHub due to their size.