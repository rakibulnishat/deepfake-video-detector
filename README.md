# Deepfake Video Detector

> AI-powered deepfake video detection using a novel **Spatiotemporal Adapter (STA)** architecture combining EfficientNet-B4 and Transformer encoding.

[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face-orange)](https://huggingface.co/spaces/nishaatt/deepfake-detector)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Live Demo

Try it here: **https://huggingface.co/spaces/nishaatt/deepfake-detector**

Upload any video and the model will return:
- FAKE / REAL verdict
- Fake probability score
- Confidence level
- 8-frame visual preview

---

## Overview

This project is **Project 01** from a 15-project AI/ML for Digital Safety research plan targeting A* conference venues (CVPR, ICCV, ICASSP).

**Problem:** Deepfake detection models trained on one dataset collapse when tested on out-of-distribution data. Commercial detectors reach only ~78% accuracy on real-world content.

**Solution:** A plug-and-play Spatiotemporal Adapter (STA) that decomposes video features into orthogonal spatial and temporal subspaces, forcing the model to learn manipulation-agnostic representations.

---

## Architecture

```
Video Input
    │
    ▼
┌─────────────────────────────────────────┐
│         Frame Sampling (8 frames)        │
└─────────────────────────────────────────┘
    │
    ├──────────────────────┐
    ▼                      ▼
┌──────────────┐    ┌──────────────────────┐
│ EfficientNet │    │ Transformer Encoder  │
│     B4       │    │  (Temporal Analysis) │
│  (Spatial)   │    │                      │
└──────────────┘    └──────────────────────┘
    │                      │
    └──────────┬───────────┘
               ▼
    ┌─────────────────────┐
    │  Spatiotemporal     │
    │  Adapter (STA)      │
    │  Orthogonal Loss    │
    └─────────────────────┘
               │
    ┌─────────────────────┐
    │  Contrastive        │
    │  Feature Bank       │
    │  (Real/Fake Proto)  │
    └─────────────────────┘
               │
               ▼
         FAKE / REAL
```

### Key Components

| Component | Description |
|-----------|-------------|
| **SpatialEncoder** | EfficientNet-B4 pretrained on ImageNet, extracts per-frame artifact features |
| **TemporalEncoder** | 2-layer Transformer with positional embedding, captures cross-frame inconsistencies |
| **SpatiotemporalAdapter** | Decomposes fused features into orthogonal spatial/temporal subspaces |
| **ContrastiveFeatureBank** | Maintains 512 real/fake prototype vectors, enables contrastive loss |
| **Classifier** | MLP with dropout, binary FAKE/REAL output |

### Loss Function

```
Total Loss = BCE Loss
           + λ₁ × Orthogonal Subspace Loss
           + λ₂ × Contrastive Loss

λ₁ = 0.1, λ₂ = 0.1
```

---

## Dataset

Trained on the **DFDC (Deepfake Detection Challenge)** dataset by Meta/Kaggle:
- 119,197 video clips
- 66 paid actors
- Diverse race, pose, and lighting conditions

**Access:** https://www.kaggle.com/competitions/deepfake-detection-challenge

---

## Project Structure

```
deepfake-video-detector/
├── app.py              ← FastAPI web app (Hugging Face deployment)
├── model.py            ← Full model architecture
├── dataset.py          ← DFDC data loading and preprocessing
├── train.py            ← Training loop with AMP and W&B logging
├── evaluate.py         ← AUC-ROC, F1, EER evaluation
├── config.yaml         ← All hyperparameters
├── Dockerfile          ← Docker deployment config
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-video-detector
cd deepfake-video-detector
pip install torch==2.1.0 torchvision==0.16.0 timm opencv-python-headless Pillow numpy fastapi uvicorn
```

---

## Training

### 1. Get the dataset
Accept the rules and attach the DFDC dataset on Kaggle:
https://www.kaggle.com/competitions/deepfake-detection-challenge

### 2. Configure training
Edit `config.yaml`:
```yaml
data:
  root: "/kaggle/input/competitions/deepfake-detection-challenge/train_sample_videos"
  num_parts: 5
  frames_per_video: 8

training:
  epochs: 10
  batch_size: 16
  learning_rate: 0.0001
```

### 3. Run training
```python
from dataset import build_dataloaders
from model import build_model
from train import train
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, val_loader = build_dataloaders(cfg, device)
model = build_model(cfg, device)
best_ckpt = train(cfg, model, train_loader, val_loader, device)
```
## Download Trained Model

The trained model weights are hosted on Hugging Face:

🔗 https://huggingface.co/nishaatt/deepfake-detector/resolve/main/best_model.pt

To download:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="nishaatt/deepfake-detector",
    filename="best_model.pt"
)
```
---

## Evaluation

```python
from evaluate import evaluate
metrics = evaluate(model, val_loader, device)
# Returns: AUC-ROC, F1, EER, Precision, Recall, Accuracy
```

### Results (sample data, 400 videos)

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.6383 |
| F1-Score | 0.8762 |
| Val Accuracy | 0.7833 |

> Note: Trained on sample data only (~400 videos). Full DFDC dataset (119K videos) expected to achieve AUC > 0.85.

---

## Deployment

### Local
```bash
python app.py
# Visit http://localhost:7860
```

### Docker
```bash
docker build -t deepfake-detector .
docker run -p 7860:7860 deepfake-detector
```

### Hugging Face Spaces
Live at: https://huggingface.co/spaces/nishaatt/deepfake-detector

---

## Baselines to Beat

| Model | Cross-Dataset AUC |
|-------|------------------|
| XceptionNet (Rossler 2019) | ~0.74 |
| EfficientNet-B4 vanilla | ~0.77 |
| CLIP-based ViT (2025) | ~0.80 |
| **This model (target)** | **> 0.85** |

---

## Ethical Safeguards

- Training uses only consented actor datasets (DFDC paid actors)
- Mandatory demographic bias audit across gender and ethnicity
- No victim material ever used
- Model outputs are advisory signals, not final determinations

---

## Research Context

This is Project 01 of a 15-project AI/ML for Digital Safety research plan.

**Target venues:** CVPR · ICCV · ECCV · IEEE TIFS · IEEE TPAMI

**Related projects:**
- Project 02: Audio Deepfake Detection (Interspeech)
- Project 13: Multilingual Harm Detection for South Asian Platforms (ACL)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{deepfake-detector-2026,
  author = {Rakibul Hassan Nishat},
  title  = {Deepfake Video Detection via Spatiotemporal Adapter},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/deepfake-video-detector}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Rakibul Hassan Nishat**
- Kaggle: [rakibulhassannishat](https://www.kaggle.com/rakibulhassannishat)
- Hugging Face: [nishaatt](https://huggingface.co/nishaatt)
