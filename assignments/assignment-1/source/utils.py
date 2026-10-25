import os
import sys
import platform
import random
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_environment(seed=42):
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__} | Seed: {seed} | Device: {device_name}")

def plot_training_curves(histories, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for name, hist in histories.items():
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], linestyle='--', label=f"{name} Train")
        axes[0].plot(epochs, hist['val_loss'], label=f"{name} Val")
        axes[1].plot(epochs, hist['val_acc'], label=name)
        axes[2].plot(epochs, hist['val_f1'], label=name)

    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "all_models_curves.png"), dpi=150)
    plt.close()

def plot_confusion_matrix(targets, preds, classes, model_name, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, values_format='d')
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"cm_{model_name.lower()}.png"), dpi=150)
    plt.close()

def plot_prediction_examples(images, targets, preds, classes, model_name, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    correct_idx = [i for i, (t, p) in enumerate(zip(targets, preds)) if t == p][:5]
    incorrect_idx = [i for i, (t, p) in enumerate(zip(targets, preds)) if t != p][:5]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for col, idx in enumerate(correct_idx):
        img = images[idx].squeeze() * 0.5 + 0.5
        axes[0, col].imshow(img, cmap='gray')
        axes[0, col].set_title(f"True: {classes[targets[idx]]}\nPred: {classes[preds[idx]]}", color='green', fontsize=9)
        axes[0, col].axis('off')

    for col, idx in enumerate(incorrect_idx):
        img = images[idx].squeeze() * 0.5 + 0.5
        axes[1, col].imshow(img, cmap='gray')
        axes[1, col].set_title(f"True: {classes[targets[idx]]}\nPred: {classes[preds[idx]]}", color='red', fontsize=9)
        axes[1, col].axis('off')

    plt.suptitle(f"{model_name}: Top Row = Correct | Bottom Row = Incorrect")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"examples_{model_name.lower()}.png"), dpi=150)
    plt.close()

def save_summary_table(summary_rows, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "comparison_summary.csv")
    headers = ["Model", "Parameters", "Best_Val_Acc", "Best_Val_F1", "Test_Acc", "Test_Macro_F1", "Train_Time_s", "Inference_Time_s"]
    with open(filepath, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(summary_rows)