# BloodMNIST Classification

> Exploring neural network architectures on medical image classification. Compare Fully Connected Networks (FCNs) vs Convolutional Neural Networks (CNNs) with regularization techniques to understand overfitting and how to combat it.

---

## Overview

This lab demonstrates classification of **8 different types of blood cells** using the **BloodMNIST** dataset (17,092 images, 28×28 pixels). Two main approaches are explored:
1. **Experiment 1**: Fully Connected Networks (FFNNs) with increasing complexity
2. **Experiment 2**: Convolutional Neural Networks (CNNs) that preserve image structure

The key learning is how adding regularization (Batch Normalization, Dropout) improves generalization and reduces overfitting.

---

## File Structure

```
Day 2 lab/
│
├── day_2_lab.ipynb           # Main Jupyter notebook with all experiments
│
├── README.md                 # This file
```

---

## Experiment 1: Fully Connected Networks (FFNNs)

### Dataset Setup
- **BloodMNIST** is downloaded automatically from the MedMNIST library
- Images are **flattened** to 1D vectors (2,352 features = 28×28×3)
- Normalized using mean=0.5, std=0.5

### Architectures

**Step 1: Sanity Check** — Single layer model trained on 4 fixed samples to verify training loop works

**Step 2: Simple Model**
```
Input (2352) → Output (8)
```

**Step 3: Complex Model (Without Regularization)**
```
Input → 2048 → 1024 → 1024 → 512 → 256 → 256 → 64 → Output (8)
```
All layers use ReLU activation. Expected to **overfit** significantly.

**Step 4: Complex Model (With Regularization)**
```
Input → [BatchNorm → ReLU] → Dropout(0.4)
  → [BatchNorm → ReLU] → Dropout(0.3)
  → [ReLU] → Output (8)
```
Adds Batch Normalization and Dropout to reduce overfitting.

### Observations
- Simple model: **baseline performance**
- Complex without regularization: **overfits** (high train accuracy, lower validation/test)
- Complex with regularization: **improves generalization** slightly, achieving ~88% test accuracy
- The BloodMNIST dataset isn't severely imbalanced, so improvements plateau around 88%

---

## Experiment 2: Convolutional Neural Networks (CNNs)

### Dataset Setup
- Same **BloodMNIST** dataset
- Images **preserved as 3D tensors** (3×28×28) to leverage spatial information
- Same normalization: mean=0.5, std=0.5

### Architectures

**Step 1: Simple CNN**
```
Conv(3→16, 3×3) → ReLU → MaxPool(2)
→ Flatten → Linear(16×14×14 → 8)
```

**Step 2: Complex CNN (Without Regularization)**
```
Conv(3→32) → ReLU → Conv(32→64) → ReLU → MaxPool(2)
→ Conv(64→128) → ReLU → Conv(128→128) → ReLU → MaxPool(2)
→ Flatten → Linear(128×7×7 → 512) → ReLU → Linear(512 → 8)
```

**Step 3: Complex CNN (With Regularization)**
```
Conv → BatchNorm2d → ReLU
→ Conv → BatchNorm2d → ReLU → MaxPool → Dropout2d(0.2)
→ Conv → BatchNorm2d → ReLU → MaxPool → Dropout2d(0.3)
→ Flatten → Linear → ReLU → Dropout(0.5) → Output(8)
```

### Why CNNs?
- **Spatial structure matters**: Blood cell images have local patterns (edges, textures)
- **Convolutional layers** learn local features efficiently with **fewer parameters** than FFNNs
- **MaxPooling** provides translation invariance

---

## Training Details

### Common Settings
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 100
- **Batch Size**: 128
- **Data Split**: Train / Val / Test (from MedMNIST)

### Evaluation
- Monitor **training loss**, **validation loss**, and **validation accuracy** every 10 epochs
- Final metrics reported on the **test set** (held out during training)

---

## Key Learnings

1. **Regularization reduces overfitting**
   - Models without regularization fit training data perfectly but fail on unseen data
   - BatchNorm + Dropout significantly improve validation/test accuracy

2. **CNNs are more efficient for images**
   - CNNs require fewer parameters than FCNNs
   - Local spatial correlations are leveraged better

3. **Dataset quality limits performance**
   - Despite architectural improvements, accuracy plateaus around 88%
   - Suggests the dataset has inherent noise or class imbalance

4. **Sanity checks are essential**
   - Training on a tiny fixed batch verifies the training loop before scaling up
   - Catches bugs early (gradient flow, device placement, etc.)

---

## Quick Start

Make sure you have a Python 3.8+ environment with dependencies installed.

```bash
# Install dependencies
pip install -r requirements.txt

# or install manually
pip install medmnist==3.0.2
pip install torch==2.10.0
pip install torchvision==0.25.0
pip install numpy==2.4.3

# Run the notebook
jupyter notebook day_2_lab.ipynb
```

---

## Notes

- The notebook automatically downloads **BloodMNIST** on first run (takes a few moments)
- Set `seed_everything(42)` at the start ensures reproducible results
- GPU acceleration is used if available (CUDA), falls back to CPU otherwise
