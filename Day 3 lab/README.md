# Pokémon Classification with Transfer Learning

> Master transfer learning by training ResNet18 on a custom Pokémon dataset. Compare from-scratch models against pretrained architectures to understand how transfer learning dramatically accelerates training and improves accuracy on limited data.

---

## Overview

This lab classifies **Pokémon species** using the **Pokémon Classification** dataset from Kaggle. Three progressively sophisticated approaches are explored:
1. **Small MLP** — Baseline: fully connected network on flattened images
2. **Custom CNN** — Training from scratch on real image data
3. **ResNet18 Transfer Learning** — Leveraging pretrained ImageNet features

The key insight: **transfer learning** achieves high accuracy in minimal training time by reusing features learned on massive datasets.

---

## File Structure

```
Day 3 lab/
│
├── day_3_lab.ipynb           # Main Jupyter notebook with all experiments
│
├── README.md                 # This file
```

---

## Dataset

**Pokémon Classification** (Kaggle):
- Multiple Pokémon species with hundreds of images per class
- Automatically downloaded via Kaggle API (requires `kaggle.json` credentials)
- Images standardized to RGB format
- Split: **72% train / 18% validation / 10% test** (stratified by class)

---

## Experiment 1: Small MLP

### Setup
- **Image size**: 128×128 pixels, flattened to **49,152 features**
- **Normalization**: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- **Data augmentation**: None

### Architecture
```
Input (49,152) → 1024 → ReLU
  → 512 → ReLU
  → 128 → ReLU
  → Output (num_classes)
```

### Training
- **Epochs**: 50
- **Batch size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss

### Performance
- MLPs are **inefficient** for images — every pixel is a separate feature
- High parameters, slow training, often gets stuck early
- Baseline accuracy for comparison

---

## Experiment 2: Custom CNN (From Scratch)

### Setup
- **Image size**: 224×224 pixels, **preserved as 3D tensors**
- **Same normalization** as Experiment 1

### Architecture
```
Conv(3→64, 3×3) → ReLU → MaxPool(2)
→ Conv(64→128, 3×3) → ReLU → MaxPool(2)
→ Conv(128→256, 3×3) → ReLU → MaxPool(2)
→ Conv(256→512, 3×3) → ReLU → MaxPool(2)
→ Flatten → Linear(100,352 → 2048) → ReLU → Linear(2048 → num_classes)
```

### Training
- **Epochs**: 50
- **Batch size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss

### Performance
- Much **better parameter efficiency** than MLP
- Local features (edges, textures) learned automatically
- Still **slow to train** — convolutional filters learned from scratch
- Modest improvement over MLP, but plateaus without more data/epochs

---

## Experiment 3: ResNet18 Transfer Learning ⭐

### The Magic of Transfer Learning
- ResNet18 **pretrained on ImageNet** (1.2M images, 1,000 classes)
- Already knows how to detect edges, shapes, textures, objects
- **Freeze all convolutional layers** — only train the final FC layer
- Dramatically **fewer parameters** to update → faster training

### Setup
- **Image size**: 224×224 pixels
- **ImageNet normalization**: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

### Architecture
```
ResNet18 Backbone (frozen) → 512-d features
  → Custom FC layer: Linear(512 → num_classes)
```

### Training
- **Epochs**: 50 (but convergence much faster!)
- **Batch size**: 32
- **Optimizer**: Adam on **FC layer only** (lr=1e-3)
- **Loss**: CrossEntropyLoss

### Key Differences
- All `model.parameters()` frozen except `model.fc`
- Only the final classification layer is trainable
- 18× fewer parameters than training full model

### Performance
- **Rapid convergence** — high accuracy achieved in first few epochs
- Far superior to both MLP and custom CNN
- Demonstrates the **power of pretrained models** on limited datasets

---

## Visualization: Feature Maps

The notebook includes a feature map visualizer that shows what the first convolutional layer learns:

```python
def visualize_multi_image_kernels(model, loader, device, num_images=10, num_kernels=10):
    # Shows original image + first 10 kernel activations for multiple test images
```

**Insight**: Each kernel/filter learns to activate on different image features:
- Low-level: edges, textures, colors
- Can be used for interpretability — what is the model actually seeing?

---

## Training Details

### Common Settings
- **Device**: CUDA if available, otherwise CPU
- **Data Split**: Train / Val / Test (stratified)
- **Evaluation**: Loss + Accuracy on validation set every 10 epochs
- **Final Report**: Test set metrics (held out during all training)

### Hyperparameters
| Experiment | Batch Size | Learning Rate | Epochs |
|-----------|-----------|--------------|--------|
| MLP | 32 | 0.001 | 50 |
| CNN | 32 | 0.001 | 50 |
| ResNet18 | 32 | 0.001 | 50 |

---

## Key Learnings

1. **Transfer learning is powerful**
   - Pretrained models from large datasets (ImageNet) transfer remarkably well
   - Often beats custom models trained from scratch with limited data
   - Especially effective for domain shift (different dataset, similar task)

2. **MLPs are inefficient for images**
   - Treating pixels independently ignores spatial structure
   - Convolutional layers are the right architectural choice for vision

3. **CNNs learn local features**
   - Filters automatically extract edges, patterns, textures
   - More parameter-efficient than MLPs but slower than transfer learning

4. **Freezing pretrained weights**
   - Only fine-tuning the final layer is often sufficient
   - Prevents overfitting on small datasets
   - Drastically reduces training time

5. **Visualization reveals what models learn**
   - Feature maps show interpretability
   - Kernels activate on semantically meaningful patterns

---

## Quick Start

You'll need Kaggle credentials to download the Pokémon dataset.

```bash
# Install dependencies
pip install torch==2.10.0
pip install torchvision==0.25.0
pip install scikit-learn==1.8.0

# For Kaggle API
pip install kaggle

# Place your kaggle.json in ~/.kaggle/
# Download from: https://www.kaggle.com/settings/account

# Run the notebook
jupyter notebook day_3_lab.ipynb
```

### First Run
The notebook prompts you to upload `kaggle.json` via Colab's file uploader, then downloads the Pokémon dataset automatically.

---

## Notes

- **Compute**: GPU strongly recommended for CNN experiments; Colab provides free GPU
- **Transfer learning setup**: ResNet18 weights loaded from `torchvision.models` (automatic download on first use)
- **Reproducibility**: `seed_everything(42)` ensures consistent results across runs
- **Dataset size**: The lab handles class imbalance via stratified train/val/test splits
