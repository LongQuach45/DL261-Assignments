import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from utils import denormalize_imagenet, save_summary_table
from data import get_dataloaders
from models import FoodResNet50
from trainer import train_model, evaluate_test_checkpoint


def collect_test_images_subset(test_loader, max_samples=2500):
    imgs_list = []
    count = 0
    for imgs, _ in test_loader:
        imgs_list.append(imgs)
        count += imgs.size(0)
        if count >= max_samples:
            break
    return torch.cat(imgs_list, dim=0)[:max_samples]


def plot_three_tier_predictions(test_loader, targets, preds, probs, classes, model_name, save_dir="./results"):
    """
    Plots 3 rows x 5 columns as required by Proposal Section 7 & Handbook Section 3.3:
      Row 1: Confident Correct predictions (highest predicted probability on true class)
      Row 2: Difficult Correct predictions (correct prediction with lowest confidence margin)
      Row 3: Failed predictions (highest-confidence misclassifications)
    """
    os.makedirs(save_dir, exist_ok=True)
    all_imgs = collect_test_images_subset(test_loader, max_samples=len(targets))
    pred_conf = probs[np.arange(len(preds)), preds]

    correct_mask = (targets == preds)
    incorrect_mask = ~correct_mask

    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]

    # Sort correct by confidence descending (confident) and ascending (difficult-correct)
    confident_correct = correct_indices[np.argsort(-pred_conf[correct_indices])][:5]
    difficult_correct = correct_indices[np.argsort(pred_conf[correct_indices])][:5]
    confident_failures = incorrect_indices[np.argsort(-pred_conf[incorrect_indices])][:5]

    rows_data = [
        ("Confident Correct", confident_correct, "darkgreen"),
        ("Difficult Correct (Low Conf)", difficult_correct, "darkorange"),
        ("Failed Predictions", confident_failures, "crimson"),
    ]

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for r_idx, (row_label, idx_arr, color) in enumerate(rows_data):
        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            if c_idx < len(idx_arr):
                sample_i = idx_arr[c_idx]
                img_np = denormalize_imagenet(all_imgs[sample_i])
                t_name = classes[targets[sample_i]].replace("_", " ")
                p_name = classes[preds[sample_i]].replace("_", " ")
                conf_val = pred_conf[sample_i]
                ax.imshow(img_np)
                ax.set_title(
                    f"[{row_label}]\nTrue: {t_name}\nPred: {p_name} ({conf_val:.2f})",
                    color=color,
                    fontsize=8,
                )
            ax.axis("off")

    plt.suptitle(
        f"{model_name}: Correct, Difficult-Correct, and Failed Test Predictions",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"examples_{model_name.lower()}.png"), dpi=150)
    plt.close()


def _infer_error_category(cls_a, cls_b):
    pair = {cls_a, cls_b}
    if pair <= {"beef_carpaccio", "beef_tartare", "beet_salad"}:
        return "Visual similarity (raw red texture / garnish overlap)"
    if pair <= {"apple_pie", "baklava", "bread_pudding", "beignets"}:
        return "Visual similarity & presentation variation (baked pastry / powdered topping)"
    if "breakfast_burrito" in pair or "bibimbap" in pair:
        return "Multi-ingredient clutter / occlusion & framing variation"
    if "baby_back_ribs" in pair:
        return "Lighting / dark glaze contrast & close-up framing"
    return "Presentation variation / suspected training label noise"


def run_error_taxonomy_analysis(targets, preds, classes, model_name, save_dir="./results"):
    """Exports per-class precision/recall/F1 and top confused class pairs with hypothesized causes."""
    os.makedirs(save_dir, exist_ok=True)
    prec, rec, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(len(classes))), zero_division=0
    )

    per_class_rows = []
    for i, cls_name in enumerate(classes):
        per_class_rows.append({
            "Class_ID": i + 1,
            "Class_Name": cls_name.replace("_", " "),
            "Precision": round(float(prec[i]), 4),
            "Recall": round(float(rec[i]), 4),
            "F1_Score": round(float(f1[i]), 4),
            "Support": int(support[i]),
            "Error_Rate": round(1.0 - float(rec[i]), 4),
        })

    per_class_path = os.path.join(save_dir, f"per_class_metrics_{model_name.lower()}.csv")
    with open(per_class_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Class_ID", "Class_Name", "Precision", "Recall", "F1_Score", "Support", "Error_Rate"],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    cm = confusion_matrix(targets, preds, labels=list(range(len(classes))))
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "True_Class": classes[i].replace("_", " "),
                    "Predicted_Class": classes[j].replace("_", " "),
                    "Count": int(cm[i, j]),
                    "Rate_Within_True_Class": round(float(cm[i, j]) / max(int(cm[i].sum()), 1), 4),
                    "Hypothesized_Error_Category": _infer_error_category(classes[i], classes[j]),
                })

    confused_pairs.sort(key=lambda x: x["Count"], reverse=True)
    pairs_path = os.path.join(save_dir, f"top_confused_pairs_{model_name.lower()}.csv")
    with open(pairs_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "True_Class",
                "Predicted_Class",
                "Count",
                "Rate_Within_True_Class",
                "Hypothesized_Error_Category",
            ],
        )
        writer.writeheader()
        writer.writerows(confused_pairs[:15])


def compute_gradcam_heatmap(model, target_layer, input_tensor, target_class=None):
    """Safe Grad-CAM computation compatible with frozen backbones and ReLU layers."""
    model.eval()
    activations, gradients = [], []

    def fwd_hook(module, inp, out):
        out_cloned = out.clone()
        activations.append(out_cloned.detach())
        out_cloned.register_hook(lambda grad: gradients.append(grad.detach()))
        return out_cloned

    handle = target_layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad()
        inp = input_tensor.clone().detach().requires_grad_(True)
        out = model(inp)
        if target_class is None:
            target_class = int(out.argmax(dim=1).item())
        score = out[0, target_class]
        score.backward()
    finally:
        handle.remove()

    act = activations[0][0]
    grad = gradients[0][0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * act).sum(dim=0, keepdim=True).unsqueeze(0))
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, target_class


def plot_gradcam_comparison(trained_models, test_loader, classes, device, save_dir="./results", num_samples=4):
    """Generates side-by-side Grad-CAM visualizations across Custom_CNN, ResNet50_Frozen, and ResNet50_Full_Finetune."""
    os.makedirs(save_dir, exist_ok=True)
    batch_imgs, batch_labels = next(iter(test_loader))
    model_names = list(trained_models.keys())

    fig, axes = plt.subplots(num_samples, len(model_names) + 1, figsize=(4 * (len(model_names) + 1), 3.6 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(num_samples):
        img_tensor = batch_imgs[row : row + 1].to(device)
        true_cls = batch_labels[row].item()
        rgb_img = denormalize_imagenet(batch_imgs[row])

        axes[row, 0].imshow(rgb_img)
        axes[row, 0].set_title(f"Original\nTrue: {classes[true_cls].replace('_', ' ')}", fontsize=9, fontweight="bold")
        axes[row, 0].axis("off")

        for col, m_name in enumerate(model_names, start=1):
            model = trained_models[m_name]
            target_layer = model.get_gradcam_target_layer()
            cam, pred_cls = compute_gradcam_heatmap(model, target_layer, img_tensor)

            axes[row, col].imshow(rgb_img)
            axes[row, col].imshow(cam, cmap="jet", alpha=0.42)
            color = "darkgreen" if pred_cls == true_cls else "crimson"
            axes[row, col].set_title(
                f"{m_name}\nPred: {classes[pred_cls].replace('_', ' ')}",
                fontsize=9,
                color=color,
            )
            axes[row, col].axis("off")

    plt.suptitle("Grad-CAM Spatial Attention Comparison Across Model Configurations", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gradcam_comparison.png"), dpi=150)
    plt.close()


def run_augmentation_ablation(
    with_aug_Summary_row,
    data_root="./data",
    manifest_dir="./manifests",
    results_dir="./results",
    ckpt_dir="./checkpoints",
    device=None,
    total_epochs=15,
    stage1_epochs=5,
    batch_size=64,
    seed=42,
    debug=False,
):
    """
    Secondary Controlled Ablation Study for M3:
    Compares Full Fine-Tuned ResNet-50 WITH training augmentation vs. WITHOUT training augmentation
    under identical split manifests, seed, and two-stage schedule.
    """
    print("\n=== Running Secondary Ablation Study: Training Data Augmentation (With vs. Without) ===")
    train_loader_noaug, val_loader_noaug, test_loader_noaug, _ = get_dataloaders(
        data_root=data_root,
        manifest_dir=manifest_dir,
        results_dir=results_dir,
        batch_size=batch_size,
        seed=seed,
        use_augmentation=False,
        debug=debug,
    )

    model_noaug = FoodResNet50(num_classes=10, pretrained=True, freeze_backbone=True)
    train_info = train_model(
        model=model_noaug,
        model_name="ResNet50_Finetune_NoAug",
        train_loader=train_loader_noaug,
        val_loader=val_loader_noaug,
        device=device,
        total_epochs=total_epochs,
        lr=1e-3,
        two_stage=True,
        stage1_epochs=stage1_epochs,
        stage2_backbone_lr=1e-4,
        stage2_head_lr=5e-4,
        checkpoint_dir=ckpt_dir,
    )
    test_info = evaluate_test_checkpoint(
        model_noaug, train_info["best_ckpt_path"], test_loader_noaug, device, batch_size=batch_size
    )

    no_aug_row = {
        "Configuration": "ResNet50_Full_Finetune (Without Augmentation)",
        "Total_Params": train_info["total_params"],
        "Trainable_Params": train_info["trainable_params"],
        "Best_Val_Acc": round(train_info["best_val_acc"], 4),
        "Best_Val_F1": round(train_info["best_val_f1"], 4),
        "Test_Acc": round(test_info["test_acc"], 4),
        "Test_Macro_F1": round(test_info["test_f1"], 4),
        "Train_Time_s": round(train_info["train_time_s"], 2),
    }

    with_aug_row_formatted = {
        "Configuration": "ResNet50_Full_Finetune (With Augmentation)",
        "Total_Params": with_aug_Summary_row["Total_Params"],
        "Trainable_Params": with_aug_Summary_row["Trainable_Params"],
        "Best_Val_Acc": with_aug_Summary_row["Best_Val_Acc"],
        "Best_Val_F1": with_aug_Summary_row["Best_Val_F1"],
        "Test_Acc": with_aug_Summary_row["Test_Acc"],
        "Test_Macro_F1": with_aug_Summary_row["Test_Macro_F1"],
        "Train_Time_s": with_aug_Summary_row["Train_Time_s"],
    }

    save_summary_table(
        [with_aug_row_formatted, no_aug_row],
        filename="ablation_augmentation_summary.csv",
        save_dir=results_dir,
    )