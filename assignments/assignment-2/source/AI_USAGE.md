# AI Usage Disclosure — Assignment 2 (CO3133)
---

## Record 1: Pre-Split File Hash Auditing & EDA Histogram Formatting (M2 Draft)
- **Tool name and version:** Google Gemini (2026 release)
- **Member who used it:** Member 1 (Quách Kiến Long)
- **Time or development stage:** Week 9–10 (M2 Draft stage)
- **Purpose:** Generating Python `hashlib.sha256` chunked file-reading boilerplate and Matplotlib 1x3 subplot formatting for plotting image width, height, and aspect-ratio distributions during Exploratory Data Analysis.
- **Affected assignment section(s):** `data.py` (`compute_file_sha256`, `build_or_load_manifests`), Report Section 1.3 & 1.4 (EDA & Split Integrity)
- **Representative prompt example:** "Write a helper function in Python to compute SHA-256 of image files in chunks and plot a 1x3 matplotlib histogram of image widths, heights, and aspect ratios."
- **How AI output was edited and verified:** Integrated the SHA-256 helper with our custom PIL `img.verify()` check and 64-bit `aHash` perceptual duplicate detector before executing `StratifiedShuffleSplit(test_size=0.10, random_state=42)`. Verified that the generated `manifests/split_manifest.csv` contained exactly 6,750 training, 750 validation, and 2,500 test records (750/75/250 per class).
- **Member responsible for final verification:** Member 1 (Quách Kiến Long)
- **Sources used for verification:** Python standard library `hashlib` and `PIL.Image` documentation, `scikit-learn` `StratifiedShuffleSplit` reference, and direct inspection of `./manifests/split_manifest.csv` and `./results/eda_class_counts.csv`.

---

## Record 2: CUDA Event Synchronization & Warmed-Up Latency Logging Boilerplate (M2 Draft)
- **Tool name and version:** Google Gemini (2026 release)
- **Member who used it:** Member 2 (Lê Quốc Bình)
- **Time or development stage:** Week 10 (M2 Draft stage)
- **Purpose:** Consulting syntax for `torch.cuda.synchronize()`, `torch.cuda.max_memory_allocated()`, and `argparse` CLI flags to log peak GPU VRAM and warmed-up batch-64 inference latency cleanly alongside training metrics.
- **Affected assignment section(s):** `trainer.py` (`measure_warmed_up_batch_latency`), `main.py` (`parse_args`), Report Section 2.3 & 3.2 (Compute-Cost Analysis)
- **Representative prompt example:** "How to accurately time warmed-up inference latency for a fixed batch of 64 images on CUDA in PyTorch and record peak allocated GPU memory in MB."
- **How AI output was edited and verified:** Placed `torch.cuda.reset_peak_memory_stats(device)` at the start of each model's training run and added 5 warm-up forward passes plus `torch.cuda.synchronize()` barriers before and after the 20 timed forward passes. Verified that the two-stage `ResNet50_Full_Finetune` parameter group transition at Epoch 6 (`20,490` $\rightarrow$ `23,528,522` trainable parameters) and AMP `GradScaler` updates functioned accurately on our RTX 5060 GPU.
- **Member responsible for final verification:** Member 2 (Lê Quốc Bình)
- **Sources used for verification:** Official PyTorch CUDA semantics (`torch.cuda` and `torch.amp`) documentation and local terminal execution logs.

---

## Record 3: Grad-CAM Heatmap Overlay & Vertical Stage-Transition Line Plotting (M3 Final)
- **Tool name and version:** Google Gemini (2026 release)
- **Member who used it:** Member 3 (Lê Thanh Lâm)
- **Time or development stage:** Week 11 (M3 Final stage)
- **Purpose:** Formatting Matplotlib `imshow` alpha blending (`cmap='jet', alpha=0.42`) to overlay 2D Grad-CAM spatial attention maps onto ImageNet-denormalized RGB images, and adding a vertical dashed line at Epoch 5 on the learning curve plots to mark the Stage $1 \rightarrow 2$ backbone unfreezing point.
- **Affected assignment section(s):** `analysis.py` (`plot_gradcam_comparison`), `utils.py` (`plot_training_curves`), Report Section 3.1 & 3.4
- **Representative prompt example:** "Overlay a normalized 224x224 heatmap over an RGB image in matplotlib with alpha blending and add a vertical dashed transition marker on a 1x3 training curve subplot."
- **How AI output was edited and verified:** Implemented tensor-level forward activation hooks (`out.clone().register_hook(...)`) inside `compute_gradcam_heatmap` so backward gradients could be captured cleanly even when the ResNet-50 backbone parameters had `requires_grad=False` in the frozen baseline. Visually verified the resulting heatmaps in `./results/gradcam_comparison.png`.
- **Member responsible for final verification:** Member 3 (Lê Thanh Lâm)
- **Sources used for verification:** PyTorch `nn.Module.register_forward_hook` and `Tensor.register_hook` documentation, Selvaraju et al. (ICCV 2017) Grad-CAM formulation, and visual verification on Food-101 test samples.

---

## Record 4: Three-Tier Prediction Grid Layout & LaTeX Report Typesetting (M3 Final Report)
- **Tool name and version:** Google Gemini (2026 release)
- **Member who used it:** Member 4 (Tô Nhã Vy)
- **Time or development stage:** Week 11–12 (M3 Final report preparation)
- **Purpose:** Formatting the 3x5 Matplotlib subplot grid for displaying Confident Correct, Difficult Correct, and Failed predictions with color-coded titles, exporting per-class error rate CSV tables, and converting CSV summary tables into LaTeX `booktabs` format with grammar proofreading.
- **Affected assignment section(s):** `analysis.py` (`plot_three_tier_predictions`, `run_error_taxonomy_analysis`), `report.tex`, `README.md`, Report Section 3.3 & 3.5 (Qualitative Results & Error Analysis)
- **Representative prompt example:** "Arrange a 3x5 matplotlib figure showing top-5 highest-confidence correct, lowest-confidence correct, and highest-confidence wrong predictions with colored titles, and format our CSV metrics as a LaTeX booktabs table."
- **How AI output was edited and verified:** Verified the `np.argsort` indexing logic on softmax probabilities so that Row 1 strictly selects high-confidence correct predictions, Row 2 selects borderline correct predictions (lowest confidence margin), and Row 3 selects high-confidence errors. Cross-checked every cell in `report.tex` against `./results/comparison_summary.csv`, `./results/ablation_augmentation_summary.csv`, and `./results/top_confused_pairs_*.csv`.
- **Member responsible for final verification:** Member 4 (Tô Nhã Vy)
- **Sources used for verification:** `scikit-learn` `precision_recall_fscore_support` documentation, generated CSV logs in `./results/`, and Overleaf PDF compilation checks.