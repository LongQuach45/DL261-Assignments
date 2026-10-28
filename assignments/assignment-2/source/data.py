import os
import csv
import json
import hashlib
from collections import Counter, defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from utils import IMAGENET_MEAN, IMAGENET_STD, denormalize_imagenet

# Predefined 10-class subset: first 10 Food-101 classes alphabetically
SELECTED_CLASSES = [
    "apple_pie",
    "baby_back_ribs",
    "baklava",
    "beef_carpaccio",
    "beef_tartare",
    "beet_salad",
    "beignets",
    "bibimbap",
    "bread_pudding",
    "breakfast_burrito",
]

FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/food-101.tar.gz"
FOOD101_ARCHIVE = "food-101.tar.gz"


def compute_file_sha256(filepath, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_ahash(pil_img, hash_size=8):
    """Compute a 64-bit perceptual average hash (hex string) from a PIL image."""
    small = pil_img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.float32)
    bits = (pixels >= pixels.mean()).flatten()
    bit_str = "".join("1" if b else "0" for b in bits)
    return f"{int(bit_str, 2):016x}"


def get_transforms(img_size=224, use_augmentation=True):
    """Returns (train_transform, eval_transform) matching Section 5 of the proposal."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


class Food101SubsetDataset(Dataset):
    """Dataset backed by a deterministic CSV manifest row list."""
    def __init__(self, records, data_root, transform=None):
        self.records = records
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.data_root, rec["relative_path"])
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        return rgb_img, int(rec["class_id"])


def build_or_load_manifests(
    data_root="./data",
    manifest_dir="./manifests",
    results_dir="./results",
    seed=42,
    force_rebuild=False,
):
    """
    Downloads Food-101 if needed, audits files (dimensions, channels, SHA-256, perceptual aHash),
    builds the deterministic 90/10 stratified split manifest (6,750 train / 750 val / 2,500 test),
    and saves EDA tables and plots.
    """
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    manifest_path = os.path.join(manifest_dir, "split_manifest.csv")
    if os.path.exists(manifest_path) and not force_rebuild:
        print(f"Loading existing deterministic split manifest from {manifest_path}")
        with open(manifest_path, mode="r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        return records

    print("Ensuring official Food-101 dataset is available in ./data ...")
    train_raw = torchvision.datasets.Food101(root=data_root, split="train", download=True)
    test_raw = torchvision.datasets.Food101(root=data_root, split="test", download=True)

    archive_path = os.path.join(data_root, FOOD101_ARCHIVE)
    archive_sha256 = compute_file_sha256(archive_path) if os.path.exists(archive_path) else "extracted_directory_only"

    class_to_sub_idx = {cls_name: i for i, cls_name in enumerate(SELECTED_CLASSES)}
    raw_Index_to_name = {i: name for i, name in enumerate(train_raw.classes)}

    def collect_split_candidates(tv_dataset, orig_split_name):
        items = []
        for img_path, raw_label in zip(tv_dataset._image_files, tv_dataset._labels):
            cls_name = raw_Index_to_name[raw_label]
            if cls_name in class_to_sub_idx:
                rel_path = os.path.relpath(img_path, data_root)
                items.append((rel_path, cls_name, class_to_sub_idx[cls_name], orig_split_name))
        return items

    candidates = collect_split_candidates(train_raw, "official_train") + collect_split_candidates(test_raw, "test")

    print(f"Auditing {len(candidates)} images across {len(SELECTED_CLASSES)} predefined classes...")
    valid_train_candidates = []
    valid_test_records = []
    excluded_records = []
    perceptual_buckets = defaultdict(list)
    seen_sha256 = {}

    widths, heights, aspect_ratios, max_dims = [], [], [], []
    channel_modes = Counter()
    sample_image_per_class = {}

    for rel_path, cls_name, cls_id, orig_split in candidates:
        full_path = os.path.join(data_root, rel_path)
        try:
            sha256_hex = compute_file_sha256(full_path)
            with Image.open(full_path) as img:
                img.verify()
            with Image.open(full_path) as img:
                mode = img.mode
                w, h = img.size
                ahash_hex = compute_ahash(img)
                if cls_name not in sample_image_per_class and orig_split == "official_train":
                    sample_image_per_class[cls_name] = img.convert("RGB").copy()
        except Exception as e:
            excluded_records.append({
                "relative_path": rel_path,
                "class_name": cls_name,
                "original_split": orig_split,
                "reason": f"corrupted_or_unreadable: {str(e)}",
            })
            continue

        channel_modes[mode] += 1
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(round(w / max(h, 1), 4))
        max_dims.append(max(w, h))

        # Check exact file hash duplicate
        if sha256_hex in seen_sha256:
            excluded_records.append({
                "relative_path": rel_path,
                "class_name": cls_name,
                "original_split": orig_split,
                "reason": f"exact_sha256_duplicate_of_{seen_sha256[sha256_hex]}",
            })
            continue
        seen_sha256[sha256_hex] = rel_path
        perceptual_buckets[ahash_hex].append((rel_path, cls_name, orig_split))

        rec = {
            "relative_path": rel_path,
            "class_name": cls_name,
            "class_id": cls_id,
            "split": orig_split,
            "seed": seed,
            "width": w,
            "height": h,
            "mode": mode,
            "sha256": sha256_hex,
            "ahash": ahash_hex,
        }
        if orig_split == "official_train":
            valid_train_candidates.append(rec)
        else:
            rec["split"] = "test"
            valid_test_records.append(rec)

    # Stratified 90/10 split of official_train -> train (6,750) and val (750)
    y_train_cand = [r["class_id"] for r in valid_train_candidates]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y_train_cand)), y_train_cand))

    for i in train_idx:
        valid_train_candidates[i]["split"] = "train"
    for i in val_idx:
        valid_train_candidates[i]["split"] = "val"

    all_records = valid_train_candidates + valid_test_records

    # Save deterministic split manifest
    manifest_fields = [
        "relative_path", "class_name", "class_id", "split", "seed",
        "width", "height", "mode", "sha256", "ahash"
    ]
    with open(manifest_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(all_records)

    # Save excluded records log
    excl_path = os.path.join(manifest_dir, "excluded_records.csv")
    with open(excl_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["relative_path", "class_name", "original_split", "reason"]
        )
        writer.writeheader()
        writer.writerows(excluded_records)

    # Save perceptual near-duplicate pairs log
    dup_pairs = []
    for ahash_val, group in perceptual_buckets.items():
        if len(group) > 1:
            for idx_a in range(len(group)):
                for idx_b in range(idx_a + 1, len(group)):
                    dup_pairs.append({
                        "ahash": ahash_val,
                        "path_1": group[idx_a][0],
                        "class_1": group[idx_a][1],
                        "split_1": group[idx_a][2],
                        "path_2": group[idx_b][0],
                        "class_2": group[idx_b][1],
                        "split_2": group[idx_b][2],
                    })
    dup_path = os.path.join(manifest_dir, "perceptual_duplicates.csv")
    with open(dup_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ahash", "path_1", "class_1", "split_1", "path_2", "class_2", "split_2"]
        )
        writer.writeheader()
        writer.writerows(dup_pairs)

    # Save EDA class counts table
    class_split_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for r in all_records:
        class_split_counts[r["class_name"]][r["split"]] += 1

    eda_counts_rows = []
    for cls_id, cls_name in enumerate(SELECTED_CLASSES):
        c = class_split_counts[cls_name]
        eda_counts_rows.append({
            "Class_ID": cls_id + 1,
            "Class_Name": cls_name.replace("_", " "),
            "Train": c["train"],
            "Val": c["val"],
            "Test": c["test"],
            "Total": c["train"] + c["val"] + c["test"],
        })
    with open(os.path.join(results_dir, "eda_class_counts.csv"), mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Class_ID", "Class_Name", "Train", "Val", "Test", "Total"])
        writer.writeheader()
        writer.writerows(eda_counts_rows)

    # Save EDA dimension & integrity summary
    dim_stats = {
        "total_audited_images": len(candidates),
        "total_valid_images": len(all_records),
        "corrupted_or_exact_duplicates_excluded": len(excluded_records),
        "perceptual_near_duplicate_pairs_flagged": len(dup_pairs),
        "channel_modes": dict(channel_modes),
        "width_min": int(np.min(widths)),
        "width_median": float(np.median(widths)),
        "width_mean": round(float(np.mean(widths)), 2),
        "width_max": int(np.max(widths)),
        "height_min": int(np.min(heights)),
        "height_median": float(np.median(heights)),
        "height_mean": round(float(np.mean(heights)), 2),
        "height_max": int(np.max(heights)),
        "aspect_ratio_min": round(float(np.min(aspect_ratios)), 3),
        "aspect_ratio_median": round(float(np.median(aspect_ratios)), 3),
        "aspect_ratio_mean": round(float(np.mean(aspect_ratios)), 3),
        "aspect_ratio_max": round(float(np.max(aspect_ratios)), 3),
        "max_side_max": int(np.max(max_dims)),
        "download_url": FOOD101_URL,
        "archive_filename": FOOD101_ARCHIVE,
        "archive_sha256": archive_sha256,
        "imagenet_mean": IMAGENET_MEAN.tolist(),
        "imagenet_std": IMAGENET_STD.tolist(),
        "seed": seed,
    }
    with open(os.path.join(manifest_dir, "reproducibility_metadata.json"), mode="w", encoding="utf-8") as f:
        json.dump(dim_stats, f, indent=2)

    # Plot representative class samples (2x5 grid)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for i, cls_name in enumerate(SELECTED_CLASSES):
        ax = axes.flatten()[i]
        if cls_name in sample_image_per_class:
            ax.imshow(sample_image_per_class[cls_name])
        ax.set_title(cls_name.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Food-101 10-Class Subset: Representative Training Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "eda_samples.png"), dpi=150)
    plt.close()

    # Plot image dimension & aspect ratio distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(widths, bins=25, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_title("Image Width Distribution (px)")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Count")

    axes[1].hist(heights, bins=25, color="teal", edgecolor="black", alpha=0.8)
    axes[1].set_title("Image Height Distribution (px)")
    axes[1].set_xlabel("Height")

    axes[2].hist(aspect_ratios, bins=30, color="coral", edgecolor="black", alpha=0.8)
    axes[2].set_title("Aspect Ratio (W / H) Distribution")
    axes[2].set_xlabel("Aspect Ratio")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "eda_dimensions.png"), dpi=150)
    plt.close()

    print(
        f"EDA & Manifest Complete: Train={sum(r['split']=='train' for r in all_records)}, "
        f"Val={sum(r['split']=='val' for r in all_records)}, "
        f"Test={sum(r['split']=='test' for r in all_records)} | "
        f"Excluded={len(excluded_records)} | Near-duplicate pairs={len(dup_pairs)}"
    )
    return all_records


def _stratified_subsample(records, per_class_limit, seed=42):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[int(r["class_id"])].append(r)
    subsampled = []
    for cls_id in sorted(by_class.keys()):
        cls_recs = by_class[cls_id]
        chosen_idx = rng.choice(len(cls_recs), size=min(per_class_limit, len(cls_recs)), replace=False)
        for idx in chosen_idx:
            subsampled.append(cls_recs[idx])
    return subsampled


def get_dataloaders(
    data_root="./data",
    manifest_dir="./manifests",
    results_dir="./results",
    img_size=224,
    batch_size=64,
    seed=42,
    use_augmentation=True,
    debug=False,
    num_workers=2,
):
    all_records = build_or_load_manifests(
        data_root=data_root, manifest_dir=manifest_dir, results_dir=results_dir, seed=seed
    )

    train_records = [r for r in all_records if r["split"] == "train"]
    val_records = [r for r in all_records if r["split"] == "val"]
    test_records = [r for r in all_records if r["split"] == "test"]

    if debug:
        # Stratified debugging subset as permitted by Proposal Section 2 & 9.1
        train_records = _stratified_subsample(train_records, per_class_limit=30, seed=seed)
        val_records = _stratified_subsample(val_records, per_class_limit=10, seed=seed)
        test_records = _stratified_subsample(test_records, per_class_limit=10, seed=seed)
        print(
            f"[DEBUG SUBSET ACTIVE] Using stratified subset (seed={seed}): "
            f"Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}"
        )

    train_tf, eval_tf = get_transforms(img_size=img_size, use_augmentation=use_augmentation)

    train_ds = Food101SubsetDataset(train_records, data_root=data_root, transform=train_tf)
    val_ds = Food101SubsetDataset(val_records, data_root=data_root, transform=eval_tf)
    test_ds = Food101SubsetDataset(test_records, data_root=data_root, transform=eval_tf)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )

    # Save one preprocessed training batch preview
    preview_path = os.path.join(results_dir, "preprocessed_batch.png")
    if not os.path.exists(preview_path):
        sample_imgs, sample_labels = next(iter(train_loader))
        print(
            f"Preprocessed batch shape: {tuple(sample_imgs.shape)} | "
            f"Min={sample_imgs.min():.2f}, Max={sample_imgs.max():.2f} | "
            f"Norm Mean={IMAGENET_MEAN.tolist()}, Std={IMAGENET_STD.tolist()}"
        )
        fig, axes = plt.subplots(1, 8, figsize=(15, 2.3))
        for i in range(min(8, len(sample_imgs))):
            img_vis = denormalize_imagenet(sample_imgs[i])
            axes[i].imshow(img_vis)
            cls_label = SELECTED_CLASSES[sample_labels[i].item()].replace("_", " ")
            axes[i].set_title(cls_label, fontsize=8)
            axes[i].axis("off")
        plt.suptitle("Preprocessed & Augmented Training Batch Slice (3 x 224 x 224)", fontsize=10)
        plt.tight_layout()
        plt.savefig(preview_path, dpi=150)
        plt.close()

    return train_loader, val_loader, test_loader, SELECTED_CLASSES