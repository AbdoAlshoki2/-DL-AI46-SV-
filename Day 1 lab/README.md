# Hand Gesture Classification

> A real-time hand gesture classifier built with MediaPipe + PyTorch. Point your hand at the camera and the model tells you what gesture you're making — instantly.

---

## Overview

Hand landmarks (21 keypoints × 3 axes = 63 values) are captured from a webcam using MediaPipe, preprocessed to be position- and scale-invariant, then fed into a small feed-forward neural network that classifies the gesture.

---

## File Structure

```
Day 1 lab/
│
├── data/
│   └── hand_landmarks_data.csv   # Training dataset (MediaPipe keypoints + label)
│
├── ml_project.py                 # Main training script — loads data, visualises, trains, saves
├── gesture_net.py                # GestureNet model definition (PyTorch nn.Module)
├── transformers.py               # Custom sklearn transformers: HandCentering, HandNormalization
├── hand_festure_demo.py          # Live webcam demo — loads the trained model and runs inference
│
├── config.yaml                   # Paths config for the demo (model + processor locations)
├── requirements.txt              # Python dependencies
│
├── processors/                   # Created after training
│   └── feature_processor.joblib  # Fitted preprocessing pipeline
│
├── models/                       # Created after training
    └── gesture_net.pth           # Model weights + class names checkpoint
```

---

## How It Works

### Preprocessing (`transformers.py`)
Two steps applied before any data touches the network:
- **HandCentering** — subtracts the wrist position so the hand location in the frame doesn't matter
- **HandNormalization** — divides by the distance to the middle finger so hand size / camera distance doesn't matter

### Model (`gesture_net.py`)
A simple 4-layer feed-forward network:
```
Input (63) → 256 → 128 → 64 → Output (num_classes)
```
Each hidden layer uses BatchNorm → ReLU → Dropout. No softmax at the end — `CrossEntropyLoss` handles that during training, and `argmax` is used at inference.

### Training (`ml_project.py`)
1. Loads and visualises the dataset
2. Splits data **70% train / 15% val / 15% test**
3. Fits the preprocessor on train only
4. Trains `GestureNet` with Adam + CosineAnnealingLR for 60 epochs
5. Monitors validation accuracy each epoch
6. Reports final metrics on the untouched test set
7. Saves `processors/feature_processor.joblib` and `models/gesture_net.pth`

### Demo (`hand_festure_demo.py`)
Opens the webcam, runs MediaPipe to detect hand landmarks, preprocesses them with the saved pipeline, and passes them through the trained model. The predicted gesture is overlaid on the video in real time.

---

## Quick Start

I usually prefer to have python 3.11 to run this code, as it is more stable with the libraries used, also I recommend creating a virtual environment for this project to avoid dependency conflicts. You can use `venv` or `conda` for that.

```bash
# Using venv
uv venv .venv --p python 3.11

# Using python -m venv
python -m venv .venv 

```

```bash
# Install dependencies (using uv)
uv pip install -r requirements.txt

# Train the model
python ml_project.py

# Run the live demo
python hand_festure_demo.py
```

