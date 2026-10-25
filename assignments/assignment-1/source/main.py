import os
import torch
from utils import (
    set_seed, count_parameters, log_environment,
    plot_training_curves, plot_confusion_matrix,
    plot_prediction_examples, save_summary_table
)
from data import run_eda, get_dataloaders
from models import (
    LinearClassifier, MLPClassifier, CNNClassifier,
    RNNClassifier, TransformerClassifier
)
from trainer import train_and_validate, evaluate_test_set
from extensions import (
    run_robustness_study,
    run_interpretability_study,
    run_matched_budget_experiment
)

def run_benchmark(dataset_name, in_channels, img_size, epochs, batch_size, lr, seed, device, save_dir, ckpt_dir):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset_name=dataset_name, batch_size=batch_size, seed=seed, save_dir=save_dir
    )

    model_builders = {
        "Linear": lambda: LinearClassifier(in_channels=in_channels, img_size=img_size, num_classes=10),
        "MLP": lambda: MLPClassifier(in_channels=in_channels, img_size=img_size, hidden_dim=256, num_classes=10),
        "CNN": lambda: CNNClassifier(in_channels=in_channels, img_size=img_size, num_classes=10),
        "LSTM": lambda: RNNClassifier(in_channels=in_channels, img_size=img_size, hidden_size=128, num_layers=2, num_classes=10, rnn_type="LSTM"),
        "GRU": lambda: RNNClassifier(in_channels=in_channels, img_size=img_size, hidden_size=128, num_layers=2, num_classes=10, rnn_type="GRU"),
        "Transformer": lambda: TransformerClassifier(img_size=img_size, patch_size=4, in_channels=in_channels, embed_dim=64, num_heads=4, num_layers=3, num_classes=10)
    }

    histories = {}
    summary_rows = []
    trained_models = {}

    for name, builder in model_builders.items():
        set_seed(seed)
        model = builder().to(device)
        params = count_parameters(model)

        print("\n" + "=" * 65)
        print(f"[{dataset_name.upper()}] Model: {name} | Parameters: {params:,}")
        print("=" * 65)

        history, train_time, best_ckpt = train_and_validate(
            model, f"{dataset_name}_{name}", train_loader, val_loader, device, epochs=epochs, lr=lr, checkpoint_dir=ckpt_dir
        )
        histories[name] = history

        test_acc, test_f1, inf_time, targets, preds, sample_imgs = evaluate_test_set(
            model, best_ckpt, test_loader, device
        )
        trained_models[name] = model

        plot_confusion_matrix(targets, preds, classes, name, save_dir=save_dir)
        plot_prediction_examples(sample_imgs, targets, preds, classes, name, save_dir=save_dir)

        summary_rows.append({
            "Model": name,
            "Parameters": params,
            "Best_Val_Acc": round(max(history['val_acc']), 4),
            "Best_Val_F1": round(max(history['val_f1']), 4),
            "Test_Acc": round(test_acc, 4),
            "Test_Macro_F1": round(test_f1, 4),
            "Train_Time_s": round(train_time, 2),
            "Inference_Time_s": round(inf_time, 2)
        })

    plot_training_curves(histories, save_dir=save_dir)
    save_summary_table(summary_rows, save_dir=save_dir)
    return trained_models, train_loader, val_loader, test_loader, classes

def main():
    seed = 42
    epochs = 10
    batch_size = 64
    lr = 1e-3

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_environment(seed)

    run_eda(save_dir="./results/fashion_mnist")

    trained_models, train_loader, val_loader, test_loader, classes = run_benchmark(
        dataset_name="fashion_mnist",
        in_channels=1,
        img_size=28,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        save_dir="./results/fashion_mnist",
        ckpt_dir="./checkpoints/fashion_mnist"
    )

    print("\nRunning Robustness & Interpretability Extensions on Fashion-MNIST...")
    run_robustness_study(trained_models, test_loader, device, save_dir="./results/fashion_mnist")
    run_interpretability_study(trained_models, test_loader, classes, device, save_dir="./results/fashion_mnist")

    print("\nRunning Matched Parameter Budget Extension on Fashion-MNIST...")
    run_matched_budget_experiment(train_loader, val_loader, test_loader, device, epochs=epochs, lr=lr, seed=seed, save_dir="./results/fashion_mnist")

    print("\nRunning CIFAR-10 Extension Benchmark...")
    run_benchmark(
        dataset_name="cifar10",
        in_channels=3,
        img_size=32,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=device,
        save_dir="./results/cifar10",
        ckpt_dir="./checkpoints/cifar10"
    )

if __name__ == "__main__":
    main()