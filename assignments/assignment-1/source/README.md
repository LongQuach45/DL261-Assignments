
```markdown
# CO3133 — Assignment 1: Foundations of Deep Learning Pipelines and Architectures

**Topic:** From Linear Models to Modern Sequence Models: A Comparative Study for Image Classification  
**Institution:** Ho Chi Minh City University of Technology (HCMUT), VNU-HCM — Faculty of Computer Science and Engineering  
**Course:** Deep Learning and Its Applications (`CO3133`) — Semester 261  
**Instructor:** Dr. Lê Thành Sách  


## 2. Project Structure

```text
BTL1/
├── checkpoints/                 # Generated automatically: stores best model weights (.pth)
│   ├── fashion_mnist/           # Best checkpoints on Fashion-MNIST (selected by Val Macro-F1)
│   └── cifar10/                 # Best checkpoints on CIFAR-10 extension
├── data/                        # Generated automatically: downloaded Torchvision datasets
├── models/                      # Self-implemented architecture families
│   ├── __init__.py              # Model exports
│   ├── linear.py                # 1. Linear / Softmax Classifier (Flattened 1D vector)
│   ├── mlp.py                   # 2. Multilayer Perceptron (BatchNorm, ReLU, Dropout)
│   ├── cnn.py                   # 3. 2-Block Convolutional Neural Network (2D spatial features)
│   ├── lstm.py                  # 4. Row-Sequence LSTM & GRU Classifiers (28 timesteps)
│   └── transformer.py           # 5. Patch Vision Transformer (4x4 patches, CLS token, PosEmbed)
├── results/                     # Generated automatically: CSV tables and PNG visualizations
│   ├── fashion_mnist/           # EDA, curves, confusion matrices, examples, robustness & saliency
│   └── cifar10/                 # Extension benchmark outputs
├── data.py                      # EDA, preprocessing transforms, and deterministic splitting
├── trainer.py                   # Training, validation checkpointing, and test evaluation loops
├── extensions.py                # Bonus studies: Robustness, Interpretability, Matched Parameter Budget
├── utils.py                     # Seed fixing, parameter counting, plotting, and CSV logging
├── main.py                      # Master entry point to reproduce all experiments end-to-end
├── requirements.txt             # Python package dependencies
├── AI_USAGE.md                  # Detailed generative AI usage disclosure and verification logs
└── README.md                    # Reproduction instructions and documentation

```

---

## 3. Environment & Hardware Specifications

To ensure strict fairness and reproducibility across all experiments, all runs were executed under the following environment:

* **Hardware Configuration:**
* **CPU:** Intel Core Ultra 9 275HX
* **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU
* **RAM:** 32 GB
* **OS:** Linux (WSL2 Kali Linux on Windows 11)


* **Software & Dependency Versions (`requirements.txt`):**
* `torch >= 2.0.0`
* `torchvision >= 0.15.0`
* `numpy >= 1.21.0`
* `matplotlib >= 3.5.0`
* `scikit-learn >= 1.0.0`



### Installation

Clone the repository, navigate to the Assignment 1 folder, and install the required dependencies:

```bash
pip install -r requirements.txt

```

---

## 4. Dataset Policy & Reproducible Splitting

* **Debugging Dataset:** None (MNIST was not used for reporting; pipeline sanity checks were run directly on Fashion-MNIST subsets).
* **Primary Benchmark Dataset:** **Fashion-MNIST** ($1 \times 28 \times 28$ grayscale images across 10 classes).
* **Optional Extension Dataset:** **CIFAR-10** ($3 \times 32 \times 32$ RGB images across 10 classes).

### Data Preparation & Splitting Protocol

You do **not** need to download the datasets manually. Running `main.py` automatically downloads and verifies both Fashion-MNIST and CIFAR-10 into `./data/` via `torchvision.datasets`.

* **Global Seed:** `seed = 42` (enforced across `random`, `numpy`, `torch`, `torch.cuda`, and `torch.Generator` with `cudnn.deterministic = True` and `cudnn.benchmark = False`).
* **Split Partition (Fashion-MNIST):**
* **Training Set:** `50,000` samples (derived from the official 60,000 training set via `random_split` with `Generator().manual_seed(42)`).
* **Validation Set:** `10,000` samples.
* **Official Test Set:** `10,000` samples (strictly held out for final checkpoint evaluation).


* **Preprocessing:** Converted to floating-point tensors in $[0.0, 1.0]$ and standardized via `Normalize((0.5,), (0.5,))` to $[-1.0, 1.0]$.

---

## 5. Training, Evaluation & Reproduction Commands

### Run Full End-to-End Pipeline (Main Benchmark + All Bonus Extensions)

Execute `main.py` from the project directory:

```bash
python main.py

```

Running this single command sequentially executes:

1. **Exploratory Data Analysis (EDA):** Prints dataset statistics and class distributions, and saves representative class samples (`eda_samples.png`) and a denormalized training batch preview (`preprocessed_batch.png`) to `./results/fashion_mnist/`.
2. **Primary Benchmark on Fashion-MNIST:** Trains and validates all 5 mandatory architectures (**Linear**, **MLP**, **CNN**, **LSTM**, **Transformer**) plus **GRU** for `10` epochs (`batch_size=64`, `Adam lr=1e-3`, `seed=42`).
3. **Checkpoint Selection & Test Evaluation:** Automatically saves the checkpoint achieving the highest **Validation Macro-F1** to `./checkpoints/fashion_mnist/<model>_best.pth`, reloads it, and evaluates Accuracy, Macro-F1, and inference latency on the 10,000-sample test split.
4. **Bonus Extension 1 — Robustness Evaluation:** Evaluates all trained models under Gaussian noise ($\sigma \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$) and $8 \times 8$ center cutout occlusion (`robustness_summary.csv`, `robustness_curve.png`).
5. **Bonus Extension 2 — Interpretability Analysis:** Extracts learned weight templates for the Linear classifier (`interp_linear_weights.png`) and computes input-gradient saliency maps across architectures (`interp_saliency.png`).
6. **Bonus Extension 3 — Matched Parameter Budget ($\approx 105\text{K}$ Params):** Trains and compares scaled variants (`MLP_105k`, `CNN_105k`, `LSTM_105k`, `GRU_105k`, `ViT_105k`) under an equal parameter budget (`matched_budget_summary.csv`).
7. **Bonus Extension 4 — CIFAR-10 Generalization:** Repeats the full training, checkpointing, and test evaluation pipeline across all architectures on CIFAR-10 (`./results/cifar10/`).

---

## 6. Checkpoint Access & Output Traceability

All reported tables and figures in our report map directly to deterministic files generated by `python main.py`:

* **Saved Model Checkpoints:**
* `./checkpoints/fashion_mnist/fashion_mnist_<model>_best.pth`
* `./checkpoints/cifar10/cifar10_<model>_best.pth`
* To evaluate an existing checkpoint without retraining, call `evaluate_test_set(model, ckpt_path, test_loader, device)` from `trainer.py`.


* **Quantitative CSV Logs:**
* `./results/fashion_mnist/comparison_summary.csv` (Main comparison table: Parameters, Val Acc, Val F1, Test Acc, Test Macro-F1, Train Time, Inference Time)
* `./results/fashion_mnist/matched_budget_summary.csv` (Matched ~105K parameter ablation)
* `./results/fashion_mnist/robustness_summary.csv` (Gaussian noise & occlusion metrics)
* `./results/cifar10/comparison_summary.csv` (CIFAR-10 extension metrics)


* **Generated Visualizations:**
* `all_models_curves.png`: Training & validation loss, accuracy, and Macro-F1 curves.
* `cm_<model>.png`: Test set confusion matrices for each architecture.
* `examples_<model>.png`: 5 correct and 5 incorrect test predictions per model.



---

## 7. Academic Integrity & AI Usage Disclosure

In compliance with Section 5 of the Course Project Handbook, all generative AI usage is documented in [`AI_USAGE.md`](https://www.google.com/search?q=./AI_USAGE.md), detailing the tool version, responsible group member, development stage, prompt summary, affected files, and verification method against official PyTorch/scikit-learn documentation and local execution logs.

```

```
