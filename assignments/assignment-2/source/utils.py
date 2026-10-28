import os
import sys
import json
import platform
import random
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_environment_info(seed=42):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    info = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "device_name": device_name,
        "gpu_total_vram_gb": vram_gb,
        "os_platform": platform.platform(),
        "seed": seed,
    }
    print(
        f"Python: {info['python_version']} | PyTorch: {info['pytorch_version']} | "
        f"Seed: {seed} | Device: {device_name} ({vram_gb} GB VRAM)"
    )
    return info


def denormalize_imagenet(img_tensor):
    """Convert a (3, H, W) ImageNet-normalized tensor/array to (H, W, 3) float in [0, 1]."""
    if hasattr(img_tensor, "cpu"):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = np.array(img_tensor, copy=True)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0.0, 1.0)


def plot_training_curves(histories, save_dir="./results", stage_transition_epoch=5):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], linestyle="--", label=f"{name} Train")
        axes[0].plot(epochs, hist["val_loss"], label=f"{name} Val")
        axes[1].plot(epochs, hist["val_acc"], marker="o", markersize=3, label=name)
        axes[2].plot(epochs, hist["val_f1"], marker="o", markersize=3, label=name)

    for idx, (title, ylabel) in enumerate([
        ("Training & Validation Loss", "Cross-Entropy Loss"),
        ("Validation Accuracy", "Accuracy"),
        ("Validation Macro-F1", "Macro-F1"),
    ]):
        if stage_transition_epoch is not None:
            axes[idx].axvline(
                x=stage_transition_epoch,
                color="gray",
                linestyle=":",
                alpha=0.8,
                label="Stage 1->2 Unfreeze" if idx == 0 else None,
            )
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(ylabel)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_models_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(targets, preds, classes, model_name, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(targets, preds)
    display_labels = [c.replace("_", " ") for c in classes]
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{model_name.lower()}.png"), dpi=150)
    plt.close()


def save_summary_table(summary_rows, filename="comparison_summary.csv", save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    if not summary_rows:
        return
    headers = list(summary_rows[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved summary table to {filepath}")