import os
import argparse
import torch

from utils import (
    set_seed,
    get_environment_info,
    plot_training_curves,
    plot_confusion_matrix,
    save_summary_table,
)
from data import get_dataloaders
from models import CustomFoodCNN, FoodResNet50
from trainer import train_model, evaluate_test_checkpoint
from analysis import (
    plot_three_tier_predictions,
    run_error_taxonomy_analysis,
    plot_gradcam_comparison,
    run_augmentation_ablation,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CO3133 Assignment 2: Food-101 10-Class Fine-Grained Image Classification (M2 & M3)"
    )
    parser.add_argument(
        "--milestone",
        type=str,
        default="all",
        choices=["m2", "m3", "all"],
        help="Milestone mode: 'm2' (EDA + baseline + frozen pretrained), 'm3' or 'all' (full controlled study + ablations)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on a small stratified debugging subset (2 epochs) to verify pipeline mechanics quickly",
    )
    parser.add_argument(
        "--skip-aug-ablation",
        action="store_true",
        help="Skip the secondary augmentation ablation study to save compute time",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed (default: 42)")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    get_environment_info(seed=args.seed)

    data_root = "./data"
    manifest_dir = "./manifests"
    results_dir = "./results_debug" if args.debug else "./results"
    ckpt_dir = "./checkpoints_debug" if args.debug else "./checkpoints"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Schedule from Section 8 of Proposal: 15 epochs total (5 Stage-1 + 10 Stage-2 for ResNet-50)
    total_epochs = 2 if args.debug else 15
    stage1_epochs = 1 if args.debug else 5

    # 1. Build/Load Deterministic Split Manifests, Run EDA, and Create DataLoaders
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_root=data_root,
        manifest_dir=manifest_dir,
        results_dir=results_dir,
        img_size=224,
        batch_size=args.batch_size,
        seed=args.seed,
        use_augmentation=True,
        debug=args.debug,
        num_workers=args.num_workers,
    )

    # 2. Define Experimental Configurations (Proposal Section 6 & 8)
    # M2 Draft requires: Simple Baseline (Custom_CNN) + Pretrained Model (ResNet50_Frozen)
    # M3 Final requires: Full Controlled Experiment (Custom_CNN vs ResNet50_Frozen vs ResNet50_Full_Finetune)
    experiment_configs = [
        {
            "name": "Custom_CNN",
            "builder": lambda: CustomFoodCNN(in_channels=3, num_classes=10, base_channels=32, dropout=0.3),
            "two_stage": False,
            "lr": 1e-3,
            "backbone_mode": "Trained from scratch",
        },
        {
            "name": "ResNet50_Frozen",
            "builder": lambda: FoodResNet50(num_classes=10, pretrained=True, freeze_backbone=True),
            "two_stage": False,
            "lr": 1e-3,
            "backbone_mode": "Frozen pretrained backbone (head only)",
        },
    ]

    if args.milestone in ("m3", "all"):
        experiment_configs.append({
            "name": "ResNet50_Full_Finetune",
            "builder": lambda: FoodResNet50(num_classes=10, pretrained=True, freeze_backbone=True),
            "two_stage": True,
            "lr": 1e-3,
            "backbone_mode": "2-Stage: 5 ep frozen head + 10 ep full fine-tune",
        })

    histories = {}
    summary_rows = []
    trained_models = {}

    for cfg in experiment_configs:
        set_seed(args.seed)
        model = cfg["builder"]()
        train_stats = train_model(
            model=model,
            model_name=cfg["name"],
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            total_epochs=total_epochs,
            lr=cfg["lr"],
            two_stage=cfg["two_stage"],
            stage1_epochs=stage1_epochs,
            stage2_backbone_lr=1e-4,
            stage2_head_lr=5e-4,
            checkpoint_dir=ckpt_dir,
        )

        histories[cfg["name"]] = train_stats["history"]

        test_stats = evaluate_test_checkpoint(
            model=model,
            ckpt_path=train_stats["best_ckpt_path"],
            test_loader=test_loader,
            device=device,
            batch_size=args.batch_size,
        )
        trained_models[cfg["name"]] = model

        print(
            f"[{cfg['name']} Final Test] Acc: {test_stats['test_acc']:.4f} | "
            f"Macro-F1: {test_stats['test_f1']:.4f} | "
            f"Warmed-up Latency (B=64): {test_stats['latency_ms_per_batch64']:.2f} ms/batch "
            f"({test_stats['latency_ms_per_img']:.3f} ms/img)"
        )

        # Save confusion matrix, 3-tier qualitative examples, and error taxonomy CSVs
        plot_confusion_matrix(
            test_stats["targets"], test_stats["preds"], classes, cfg["name"], save_dir=results_dir
        )
        plot_three_tier_predictions(
            test_loader,
            test_stats["targets"],
            test_stats["preds"],
            test_stats["probs"],
            classes,
            cfg["name"],
            save_dir=results_dir,
        )
        run_error_taxonomy_analysis(
            test_stats["targets"], test_stats["preds"], classes, cfg["name"], save_dir=results_dir
        )

        summary_rows.append({
            "Model": cfg["name"],
            "Backbone_Policy": cfg["backbone_mode"],
            "Total_Params": train_stats["total_params"],
            "Trainable_Params": train_stats["trainable_params"],
            "Best_Epoch": train_stats["best_epoch"],
            "Best_Val_Acc": round(train_stats["best_val_acc"], 4),
            "Best_Val_F1": round(train_stats["best_val_f1"], 4),
            "Test_Acc": round(test_stats["test_acc"], 4),
            "Test_Macro_F1": round(test_stats["test_f1"], 4),
            "Train_Time_s": round(train_stats["train_time_s"], 2),
            "Test_Eval_Time_s": test_stats["test_eval_time_s"],
            "Latency_ms_Batch64": test_stats["latency_ms_per_batch64"],
            "Latency_ms_Per_Img": test_stats["latency_ms_per_img"],
            "Peak_VRAM_MB": train_stats["peak_vram_mb"],
        })

    # 3. Save Learning Curves and Quantitative Summary Tables
    plot_training_curves(histories, save_dir=results_dir, stage_transition_epoch=stage1_epochs)
    summary_filename = "m2_preliminary_summary.csv" if args.milestone == "m2" else "comparison_summary.csv"
    save_summary_table(summary_rows, filename=summary_filename, save_dir=results_dir)

    # 4. Run M3 Final Visualizations & Secondary Ablation Study
    if args.milestone in ("m3", "all"):
        print("\nGenerating Grad-CAM Spatial Attention Comparison...")
        plot_gradcam_comparison(trained_models, test_loader, classes, device, save_dir=results_dir)

        if not args.skip_aug_ablation:
            finetune_row = next(r for r in summary_rows if r["Model"] == "ResNet50_Full_Finetune")
            run_augmentation_ablation(
                with_aug_Summary_row=finetune_row,
                data_root=data_root,
                manifest_dir=manifest_dir,
                results_dir=results_dir,
                ckpt_dir=ckpt_dir,
                device=device,
                total_epochs=total_epochs,
                stage1_epochs=stage1_epochs,
                batch_size=args.batch_size,
                seed=args.seed,
                debug=args.debug,
            )

    print(f"\nAll Assignment 2 ({args.milestone.upper()}) outputs saved to {results_dir}/ and {manifest_dir}/.")


if __name__ == "__main__":
    main()