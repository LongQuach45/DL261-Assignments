import os
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from utils import set_seed, count_parameters
from models import MLPClassifier, CNNClassifier, RNNClassifier, TransformerClassifier
from trainer import train_and_validate, evaluate_test_set

def add_gaussian_noise(inputs, sigma):
    if sigma <= 0.0:
        return inputs
    noisy = inputs + torch.randn_like(inputs) * sigma
    return torch.clamp(noisy, -1.0, 1.0)

def add_cutout_occlusion(inputs, mask_size=8):
    occluded = inputs.clone()
    _, _, h, w = occluded.shape
    top = (h - mask_size) // 2
    left = (w - mask_size) // 2
    occluded[:, :, top:top+mask_size, left:left+mask_size] = -1.0
    return occluded

def evaluate_perturbed(model, test_loader, device, perturb_fn):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = perturb_fn(inputs.to(device))
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1

def run_robustness_study(trained_models, test_loader, device, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5]
    robustness_rows = []
    acc_curves = {name: [] for name in trained_models}

    for name, model in trained_models.items():
        row = {"Model": name}
        for s in sigmas:
            acc, f1 = evaluate_perturbed(model, test_loader, device, lambda x, sig=s: add_gaussian_noise(x, sig))
            row[f"Noise_{s}_Acc"] = round(acc, 4)
            row[f"Noise_{s}_F1"] = round(f1, 4)
            acc_curves[name].append(acc)

        occ_acc, occ_f1 = evaluate_perturbed(model, test_loader, device, lambda x: add_cutout_occlusion(x, 8))
        row["Occlusion_Acc"] = round(occ_acc, 4)
        row["Occlusion_F1"] = round(occ_f1, 4)
        robustness_rows.append(row)

    csv_path = os.path.join(save_dir, "robustness_summary.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(robustness_rows[0].keys()))
        writer.writeheader()
        writer.writerows(robustness_rows)

    plt.figure(figsize=(8, 5))
    for name, curve in acc_curves.items():
        plt.plot(sigmas, curve, marker='o', label=name)
    plt.title("Robustness Evaluation: Test Accuracy vs. Gaussian Noise")
    plt.xlabel("Gaussian Noise Standard Deviation (sigma)")
    plt.ylabel("Test Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "robustness_curve.png"), dpi=150)
    plt.close()

def run_interpretability_study(trained_models, test_loader, classes, device, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)

    if "Linear" in trained_models:
        weights = trained_models["Linear"].linear.weight.detach().cpu().numpy()
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for i, ax in enumerate(axes.flatten()):
            w_img = weights[i].reshape(28, 28)
            ax.imshow(w_img, cmap='RdBu_r')
            ax.set_title(classes[i])
            ax.axis('off')
        plt.suptitle("Linear Classifier Learned Weight Templates")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "interp_linear_weights.png"), dpi=150)
        plt.close()

    sample_imgs, sample_targets = next(iter(test_loader))
    img = sample_imgs[0:1].to(device)
    target_idx = sample_targets[0].item()

    model_names = [m for m in ["MLP", "CNN", "LSTM", "Transformer"] if m in trained_models]
    fig, axes = plt.subplots(1, len(model_names) + 1, figsize=(3 * (len(model_names) + 1), 3.2))

    axes[0].imshow(img.detach().cpu().squeeze() * 0.5 + 0.5, cmap='gray')
    axes[0].set_title(f"Input: {classes[target_idx]}")
    axes[0].axis('off')

    for idx, name in enumerate(model_names):
        model = trained_models[name]
        model.eval()
        if hasattr(model, "rnn"):
            model.rnn.train()

        inp = img.clone().detach().requires_grad_(True)
        model.zero_grad()
        out = model(inp)
        score = out[0, target_idx]
        score.backward()

        saliency = inp.grad.data.abs().squeeze().cpu().numpy()
        axes[idx + 1].imshow(saliency, cmap='hot')
        axes[idx + 1].set_title(f"{name} Saliency")
        axes[idx + 1].axis('off')

        model.eval()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "interp_saliency.png"), dpi=150)
    plt.close()

def run_matched_budget_experiment(train_loader, val_loader, test_loader, device, epochs=10, lr=1e-3, seed=42, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    matched_builders = {
        "MLP_105k": lambda: MLPClassifier(in_channels=1, img_size=28, hidden_dim=124, num_classes=10),          # 105,968 params
        "CNN_105k": lambda: CNNClassifier(in_channels=1, img_size=28, base_channels=12, fc_dim=86, num_classes=10), # 104,900 params
        "LSTM_105k": lambda: RNNClassifier(in_channels=1, img_size=28, hidden_size=88, num_layers=2, num_classes=10, rnn_type="LSTM"), # 105,874 params
        "GRU_105k": lambda: RNNClassifier(in_channels=1, img_size=28, hidden_size=102, num_layers=2, num_classes=10, rnn_type="GRU"),  # 104,968 params
        "ViT_105k": lambda: TransformerClassifier(img_size=28, patch_size=4, in_channels=1, embed_dim=64, num_heads=4, num_layers=3, mlp_dim=128, num_classes=10) # 105,546 params
    }

    rows = []
    for name, builder in matched_builders.items():
        set_seed(seed)
        model = builder().to(device)
        params = count_parameters(model)
        history, train_time, ckpt = train_and_validate(model, name, train_loader, val_loader, device, epochs=epochs, lr=lr)
        test_acc, test_f1, inf_time, _, _, _ = evaluate_test_set(model, ckpt, test_loader, device)
        rows.append({
            "Model": name,
            "Parameters": params,
            "Val_Acc": round(max(history['val_acc']), 4),
            "Val_F1": round(max(history['val_f1']), 4),
            "Test_Acc": round(test_acc, 4),
            "Test_Macro_F1": round(test_f1, 4),
            "Train_Time_s": round(train_time, 2),
            "Inference_Time_s": round(inf_time, 2)
        })

    csv_path = os.path.join(save_dir, "matched_budget_summary.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
