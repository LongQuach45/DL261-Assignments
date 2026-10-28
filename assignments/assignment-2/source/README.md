# CO3133 — Assignment 2: Deep Learning on Large-Scale Data and Specialized Tasks

**Topic:** Fine-Grained Food Image Classification: Custom CNN vs. Two-Stage Pretrained ResNet-50 on Food-101  
**Track:** Image Classification  
**Institution:** Ho Chi Minh City University of Technology (HCMUT), VNU-HCM — Faculty of Computer Science and Engineering  
**Course:** Deep Learning and Its Applications (`CO3133`) — Semester 261  
**Instructor:** Dr. Lê Thành Sách  


## 2. Project Structure

```text
assignment-2/source/
├── checkpoints/                 # Auto-generated: best model weights (.pth) selected by Val Macro-F1
├── data/                        # Auto-generated: official ETH Zurich Food-101 archive & extracted files
├── manifests/                   # Auto-generated: deterministic split & pre-split integrity logs
│   ├── split_manifest.csv       # Image-level train (6,750), val (750), and test (2,500) manifest
│   ├── excluded_records.csv     # Log of corrupted files or exact SHA-256 duplicates removed pre-split
│   ├── perceptual_duplicates.csv# Log of 64-bit aHash perceptual near-duplicate pairs
│   └── reproducibility_metadata.json # Archive SHA-256, URL, image dimensions, and normalization stats
├── models/                      # Self-implemented & transfer-learning model architectures
│   ├── __init__.py              # Exports CustomFoodCNN and FoodResNet50
│   ├── custom_cnn.py            # Self-implemented 4-block CNN baseline (1,242,090 params, from scratch)
│   └── resnet50.py              # ImageNet-pretrained ResNet-50 (freeze / unfreeze / differential LRs)
├── results/                     # Auto-generated: CSV tables and PNG visualizations
├── data.py                      # Food-101 10-class subset loader, SHA-256/aHash audit, EDA, transforms
├── trainer.py                   # AMP training loop, 2-stage fine-tuning, warmed-up batch-64 latency & VRAM
├── analysis.py                  # 3-tier prediction grids, error taxonomy CSVs, Grad-CAM, aug ablation
├── utils.py                     # Deterministic seeding, total/trainable param counts, plotting, CSV writer
├── main.py                      # Master CLI entry point (--milestone {m2,m3,all}, --debug)
├── requirements.txt             # Python package dependencies
├── AI_USAGE.md                  # Detailed generative AI usage disclosure and verification logs
└── README.md                    # Reproduction instructions and documentation
```

---

## 3. Environment & Hardware Specifications

All main experiments were executed under the following hardware and software environment:

* **Hardware Configuration:**
  * **CPU:** Intel Core Ultra 9 275HX
  * **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)
  * **RAM:** 32 GB
  * **OS:** Linux (WSL2 Kali Linux on Windows 11)
* **Software & Dependency Versions (`requirements.txt`):**
  * `Python 3.13.13`
  * `torch 2.12.1+cu130`
  * `torchvision >= 0.15.0`
  * `numpy >= 1.21.0`
  * `matplotlib >= 3.5.0`
  * `scikit-learn >= 1.0.0`
  * `Pillow >= 9.0.0`

### Installation
Navigate to the Assignment 2 source directory and install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Dataset Policy, Pre-Split Integrity & Leakage Prevention

* **Dataset Source:** Official ETH Zurich Food-101 release (Bossard et al., ECCV 2014).
* **Predefined 10-Class Subset:** Fixed prior to experimentation as the first 10 classes alphabetically:
  1. `apple_pie`
  2. `baby_back_ribs`
  3. `baklava`
  4. `beef_carpaccio`
  5. `beef_tartare`
  6. `beet_salad`
  7. `beignets`
  8. `bibimbap`
  9. `bread_pudding`
  10. `breakfast_burrito`
* **Pre-Split Integrity Audit:** Before splitting, `data.py` verifies PIL image readability, RGB mode conversion, exact file SHA-256 hashes, and 64-bit average perceptual hashes (`aHash`), recording any exclusions in `manifests/excluded_records.csv`.
* **Deterministic Splitting & Leakage Control (`seed = 42`):**
  * **Split Unit:** Individual image record (`manifests/split_manifest.csv`).
  * **Training Partition:** `6,750` images (90% stratified split of the 7,500 official training images).
  * **Validation Partition:** `750` images (10% stratified split of the 7,500 official training images, 75 per class).
  * **Official Test Partition:** `2,500` images (250 per class, strictly untouched until final checkpoint evaluation).
* **Preprocessing & Augmentation:**
  * **Input Resolution:** $3 \times 224 \times 224$ RGB tensors normalized with ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
  * **Training Branch:** `RandomResizedCrop(224, scale=(0.7, 1.0))`, `RandomHorizontalFlip(p=0.5)`, and `ColorJitter(0.2, 0.2, 0.2, 0.05)`.
  * **Validation/Test Branch:** Deterministic `Resize(256)` followed by `CenterCrop(224)`.

---

## 5. Training, Evaluation & Reproduction Commands

You do **not** need to download Food-101 manually. Running `main.py` automatically downloads and verifies the dataset into `./data/` (and skips downloading if already present).

### 1. Fast Debugging Sanity Check (Stratified Subset)
To verify the data loader, manifest generation, 2-stage fine-tuning transition, Grad-CAM, and plotting in ~1 minute on a small stratified subset (30 train / 10 val / 10 test per class, 2 epochs; writes to `./results_debug/` so main results are never overwritten):

```bash
python main.py --debug
```

### 2. Milestone 2 (M2 Draft: EDA + Custom CNN Baseline + Frozen ResNet-50)
Runs the full EDA audit, builds the deterministic manifest, trains `Custom_CNN` (15 epochs) and `ResNet50_Frozen` (15 epochs, head-only), and saves preliminary metrics to `./results/m2_preliminary_summary.csv`:

```bash
python main.py --milestone m2
```

### 3. Milestone 3 (M3 Final: Complete Controlled Study, Error Analysis, Grad-CAM & Ablation)
Runs the complete controlled experiment comparing:
1. **`Custom_CNN`**: Self-implemented 4-block CNN trained from scratch (`1,242,090` trainable params, 15 epochs, `AdamW lr=1e-3`).
2. **`ResNet50_Frozen`**: ImageNet-pretrained ResNet-50 with frozen backbone (`20,490` trainable head params, `23,528,522` total params, 15 epochs, `AdamW lr=1e-3`).
3. **`ResNet50_Full_Finetune`**: Two-stage fine-tuning—Stage 1 freezes the backbone for 5 epochs (`head_lr=1e-3`), and Stage 2 unfreezes all layers (`23,528,522` trainable params) for 10 epochs with differential learning rates (`backbone_lr=1e-4`, `head_lr=5e-4`).
4. **Secondary Augmentation Ablation**: Compares `ResNet50_Full_Finetune` trained **with** vs. **without** training data augmentation.

```bash
python main.py --milestone m3
```
*(Optional flags: `--skip-aug-ablation` to skip the secondary ablation run; `--batch-size 64`; `--seed 42`; `--num-workers 2`).*

---

## 6. Checkpoint Access & Output Traceability

All reported tables and figures map directly to deterministic files generated by `main.py`:

* **Model Checkpoints (selected by highest Validation Macro-F1):**
  * `./checkpoints/custom_cnn_best.pth`
  * `./checkpoints/resnet50_frozen_best.pth`
  * `./checkpoints/resnet50_full_finetune_best.pth`
  * `./checkpoints/resnet50_finetune_noaug_best.pth`
* **Quantitative CSV Logs:**
  * `./results/comparison_summary.csv`: Main comparison table (Total Params, Trainable Params, Best Epoch, Val Acc/F1, Test Acc/Macro-F1, Train Time, Test Eval Time, Warmed-up Batch-64 Latency in `ms/batch` and `ms/img`, and Peak GPU VRAM in `MB`).
  * `./results/ablation_augmentation_summary.csv`: Secondary controlled ablation results (With vs. Without Augmentation).
  * `./results/eda_class_counts.csv`: Per-class sample counts across Train, Val, and Test partitions.
  * `./results/per_class_metrics_<model>.csv`: Per-class Precision, Recall, F1, Support, and Error Rate.
  * `./results/top_confused_pairs_<model>.csv`: Top misclassified class pairs mapped to our 5-category error taxonomy.
* **Generated Visualizations:**
  * `eda_samples.png` & `eda_dimensions.png`: Representative class images and width/height/aspect-ratio distributions.
  * `preprocessed_batch.png`: Denormalized 8-sample slice of a $3 \times 224 \times 224$ training batch.
  * `all_models_curves.png`: Training/validation loss, accuracy, and Macro-F1 curves (with Stage $1 \rightarrow 2$ unfreeze marker at Epoch 5).
  * `cm_<model>.png`: 10-class test set confusion matrices.
  * `examples_<model>.png`: 3-tier qualitative prediction grids (**Row 1: Confident Correct**, **Row 2: Difficult Correct**, **Row 3: Failed Predictions**).
  * `gradcam_comparison.png`: Side-by-side Grad-CAM spatial attention overlays across all three main models.

---

## 7. Academic Integrity & AI Usage Disclosure

In compliance with Section 5 of the Course Project Handbook, all generative AI usage is documented in [`AI_USAGE.md`](./AI_USAGE.md), detailing the tool version, responsible group member, development stage, prompt summary, affected files, and verification method against official PyTorch/Torchvision/scikit-learn documentation and local execution logs.