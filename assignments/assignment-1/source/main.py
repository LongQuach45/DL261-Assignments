"""
main.py
The main entry point for running the M1 Draft experiments.
"""
import torch
from utils import set_seed
from data import run_eda, get_dataloaders
from models import LinearClassifier, MLPClassifier, CNNClassifier
from trainer import train_and_validate

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # run_eda()


    print("\nPreparing DataLoaders...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(batch_size=64)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")


    print("\n" + "="*40)
    print("Training Linear Classifier (Baseline)")
    print("="*40)
    linear_model = LinearClassifier().to(device)
    linear_history = train_and_validate(linear_model, train_loader, val_loader, device, epochs=5)

    print("\n" + "="*40)
    print("Training Multilayer Perceptron (MLP)")
    print("="*40)
    mlp_model = MLPClassifier().to(device)
    mlp_history = train_and_validate(mlp_model, train_loader, val_loader, device, epochs=5)

    print("\n" + "="*40)
    print("Training Convolutional Neural Network (CNN)")
    print("="*40)
    
    # Fashion-MNIST has 1 channel. If testing the CIFAR-10 extension, change in_channels to 3.
    cnn_model = CNNClassifier(in_channels=1).to(device)
    cnn_history = train_and_validate(cnn_model, train_loader, val_loader, device, epochs=5)
    
if __name__ == "__main__":
    main()