import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from utils import count_parameters


def evaluate_split(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    all_probs = np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 10))
    return avg_loss, acc, macro_f1, np.array(all_targets), np.array(all_preds), all_probs


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    device,
    total_epochs=15,
    lr=1e-3,
    weight_decay=1e-4,
    two_stage=False,
    stage1_epochs=5,
    stage2_backbone_lr=1e-4,
    stage2_head_lr=5e-4,
    use_amp=True,
    checkpoint_dir="./checkpoints",
):
    """
    Trains either a single-stage model (Custom_CNN or ResNet50_Frozen) or a two-stage model
    (ResNet50_Full_Finetune: 5 epochs frozen head-only + 10 epochs unfrozen full fine-tuning).
    Selects the best checkpoint strictly by highest Validation Macro-F1.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if two_stage and hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            model.get_param_groups(backbone_lr=0.0, head_lr=lr, weight_decay=weight_decay)
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_f1 = -1.0
    best_val_acc = 0.0
    best_epoch = 1
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_name.lower()}_best.pth")

    total_params, trainable_stage1 = count_parameters(model)
    trainable_peak = trainable_stage1
    print(
        f"\n--- Training {model_name} | Total Params: {total_params:,} | "
        f"Initial Trainable: {trainable_stage1:,} | AMP: {amp_enabled} ---"
    )

    start_time = time.perf_counter()

    for epoch in range(1, total_epochs + 1):
        # Transition from Stage 1 (Frozen) -> Stage 2 (Full Fine-Tuning)
        if two_stage and epoch == (stage1_epochs + 1) and hasattr(model, "unfreeze_backbone"):
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.get_param_groups(
                    backbone_lr=stage2_backbone_lr,
                    head_lr=stage2_head_lr,
                    weight_decay=weight_decay,
                )
            )
            _, trainable_stage2 = count_parameters(model)
            trainable_peak = max(trainable_peak, trainable_stage2)
            print(
                f"  [Stage 2 Transition at Epoch {epoch}] Unfroze full ResNet-50 backbone "
                f"(Trainable params: {trainable_stage2:,} | Backbone LR={stage2_backbone_lr}, Head LR={stage2_head_lr})"
            )

        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_loss, val_acc, val_f1, _, _, _ = evaluate_split(model, val_loader, criterion, device)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), best_ckpt_path)

        print(
            f"Epoch [{epoch:02d}/{total_epochs:02d}] "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time_s = time.perf_counter() - start_time
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
        if device.type == "cuda"
        else 0.0
    )

    print(
        f"Best Val Checkpoint for {model_name}: Epoch {best_epoch} "
        f"(Val Acc={best_val_acc:.4f}, Val Macro-F1={best_val_f1:.4f}) | "
        f"Train Time={train_time_s:.2f}s | Peak VRAM={peak_vram_mb:.1f} MB"
    )
    return {
        "history": history,
        "best_ckpt_path": best_ckpt_path,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "train_time_s": train_time_s,
        "peak_vram_mb": peak_vram_mb,
        "total_params": total_params,
        "trainable_params": trainable_peak,
    }


def measure_warmed_up_batch_latency(model, device, batch_size=64, img_size=224, warmup_runs=5, timed_runs=20):
    """
    Measures warmed-up inference latency (ms/batch and ms/image) for a fixed batch of 64 images
    as specified in Section 8 of the Dataset Proposal.
    """
    model.eval()
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(timed_runs):
            _ = model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    ms_per_batch = (elapsed / timed_runs) * 1000.0
    ms_per_img = ms_per_batch / batch_size
    return round(ms_per_batch, 2), round(ms_per_img, 3)


def evaluate_test_checkpoint(model, ckpt_path, test_loader, device, batch_size=64):
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    test_loss, test_acc, test_f1, targets, preds, probs = evaluate_split(model, test_loader, criterion, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_eval_time_s = round(time.perf_counter() - t0, 2)

    ms_per_batch, ms_per_img = measure_warmed_up_batch_latency(model, device, batch_size=batch_size)
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_eval_time_s": test_eval_time_s,
        "latency_ms_per_batch64": ms_per_batch,
        "latency_ms_per_img": ms_per_img,
        "targets": targets,
        "preds": preds,
        "probs": probs,
    }