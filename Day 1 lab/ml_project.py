import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from gesture_net import GestureNet
from transformers import HandCentering, HandNormalization




def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Label distribution:\n{data['label'].value_counts()}\n")
    return data



def visualize_label_distribution(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.countplot(y=data['label'], color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Labels', fontsize=16, fontweight='bold')
    plt.xlabel('Number of samples', fontsize=12)
    plt.ylabel('Label', fontsize=12)
    plt.tight_layout()
    plt.show()



def visualize_pca(features, labels: pd.Series, title: str = 'PCA Plot'):
    pca = sk.decomposition.PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, alpha=0.6)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def visualize_keypoints(features, labels: pd.Series, title_prefix: str = 'Keypoint'):
    data_array = features if isinstance(features, np.ndarray) else features.values
    fig, axes = plt.subplots(3, 7, figsize=(60, 45))
    axes = axes.flatten()

    for i in range(21):
        x_idx = i * 3
        y_idx = i * 3 + 1
        sns.scatterplot(
            x=data_array[:, x_idx],
            y=data_array[:, y_idx],
            hue=labels,
            ax=axes[i],
            legend=False,
            alpha=0.6
        )
        axes[i].set_title(f'{title_prefix} {i + 1}', fontsize=12)
        axes[i].set_xlabel(f'X position for point {i + 1}', fontsize=9)
        axes[i].set_ylabel(f'Y position for point {i + 1}', fontsize=9)

    handles, vis_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, vis_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout()
    plt.show()




def build_preprocessor() -> Pipeline:
    return Pipeline([
        ('hand_centering', HandCentering()),
        ('hand_normalization', HandNormalization()),
    ])


def analyze_variance(features: pd.DataFrame):
    print("Variance Analysis:")
    for axis, offset, label in [('Z', 2, 'z'), ('Y', 1, 'y'), ('X', 0, 'x')]:
        print(f"\n--- {axis} columns ---")
        for i in range(21):
            col_idx = i * 3 + offset
            print(f"  Variance of {label}{i} = {np.var(features.iloc[:, col_idx]):.6f}")
    print("\nNote: Z columns have near-zero variance (depth info kept for camera distance).\n")




def train_model(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_processor: Pipeline = None,
    epochs: int = 60,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    random_state: int = 42,
) -> GestureNet:
    """
    Preprocess data, build a GestureNet, train it and return the fitted model.

    Saved artefacts
    ---------------
    processors/feature_processor.joblib
    models/gesture_net.pth  (state-dict + metadata checkpoint)
    """
    torch.manual_seed(random_state)

    if feature_processor is None:
        feature_processor = build_preprocessor()

    class_names = sorted(labels.unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")


    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels,
        test_size=0.15,
        random_state=random_state,
        stratify=labels,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15 / 0.85,        
        random_state=random_state,
        stratify=y_train_val,
    )
    print(f"Split — train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")


    X_train_proc = feature_processor.fit_transform(X_train).astype(np.float32)
    X_val_proc   = feature_processor.transform(X_val).astype(np.float32)
    X_test_proc  = feature_processor.transform(X_test).astype(np.float32)


    y_train_enc = np.array([label_to_idx[c] for c in y_train], dtype=np.int64)
    y_val_enc   = np.array([label_to_idx[c] for c in y_val],   dtype=np.int64)
    y_test_enc  = np.array([label_to_idx[c] for c in y_test],  dtype=np.int64)


    os.makedirs('processors', exist_ok=True)
    joblib.dump(feature_processor, 'processors/feature_processor.joblib')
    print("Feature processor saved to processors/")

    train_ds = TensorDataset(torch.from_numpy(X_train_proc), torch.from_numpy(y_train_enc))
    val_ds   = TensorDataset(torch.from_numpy(X_val_proc),   torch.from_numpy(y_val_enc))
    test_ds  = TensorDataset(torch.from_numpy(X_test_proc),  torch.from_numpy(y_test_enc))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    model     = GestureNet(input_size=63, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_accuracies = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_batch)

        scheduler.step()
        train_losses.append(running_loss / len(train_ds))




        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                correct += (model(X_batch).argmax(dim=1) == y_batch).sum().item()
                total   += len(y_batch)
        val_accuracies.append(correct / total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{epochs}  loss={train_losses[-1]:.4f}  val_acc={val_accuracies[-1]:.4f}")



    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy())

    print("\n--- Test Set Evaluation ---")
    print(f"  Accuracy  = {accuracy_score(all_targets, all_preds):.4f}")
    print(f"  F1        = {f1_score(all_targets, all_preds, average='weighted'):.4f}")
    print(f"  Recall    = {recall_score(all_targets, all_preds, average='weighted'):.4f}")
    print(f"  Precision = {precision_score(all_targets, all_preds, average='weighted'):.4f}")


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses);   ax1.set_title('Training Loss');       ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax2.plot(val_accuracies); ax2.set_title('Validation Accuracy'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
    print("Training curves saved to artifacts/training_curves.png")

    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes':  num_classes,
        'input_size':   63,
        'class_names':  class_names,
    }, 'models/gesture_net.pth')
    print("Model checkpoint saved to models/gesture_net.pth")

    return model




def main():
    data = load_data('data/hand_landmarks_data.csv')

    features = data.drop('label', axis=1)
    labels   = data['label']

    visualize_label_distribution(data)
    visualize_pca(features, labels, title='PCA Plot Raw Features')
    visualize_keypoints(features.values, labels, title_prefix='Keypoint Raw')

    analyze_variance(features)

    preprocessor = build_preprocessor()
    new_features  = preprocessor.fit_transform(features)
    visualize_keypoints(new_features, labels, title_prefix='Keypoint Preprocessed')
    visualize_pca(new_features, labels, title='PCA Plot After Preprocessing')

    train_model(
        features=features,
        labels=labels,
        feature_processor=build_preprocessor(),
        epochs=60,
        batch_size=64,
        learning_rate=1e-3,
        random_state=42,
    )


if __name__ == '__main__':
    main()
