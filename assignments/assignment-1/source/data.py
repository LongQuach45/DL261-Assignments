import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

def run_eda(save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    eda_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    classes = eda_dataset.classes
    labels = [label for _, label in eda_dataset]
    class_counts = np.bincount(labels)

    print(f"Total training samples: {len(eda_dataset)}")
    print(f"Image shape: {eda_dataset[0][0].shape} (Channels, Height, Width)")
    print("\nClass Distribution:")
    for i, count in enumerate(class_counts):
        print(f"{classes[i]}: {count} samples")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flatten()):
        idx = labels.index(i)
        img, label = eda_dataset[idx]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(classes[label])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "eda_samples.png"), dpi=150)
    plt.close()

def get_dataloaders(dataset_name="fashion_mnist", batch_size=64, seed=42, save_dir="./results"):
    os.makedirs(save_dir, exist_ok=True)
    if dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset_class = torchvision.datasets.CIFAR10
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset_class = torchvision.datasets.FashionMNIST

    full_train_dataset = dataset_class(root='./data', train=True, download=True, transform=transform)

    if dataset_name.lower() == "cifar10":
        train_size = 45000
    else:
        train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    test_dataset = dataset_class(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    sample_imgs, sample_labels = next(iter(train_loader))
    print(f"Dataset: {dataset_name} | Split: Train={train_size}, Val={val_size}, Test={len(test_dataset)} | Seed={seed}")
    print(f"Preprocessed batch shape: {sample_imgs.shape} | Min={sample_imgs.min():.2f}, Max={sample_imgs.max():.2f}")

    fig, axes = plt.subplots(1, 8, figsize=(14, 2))
    for i in range(8):
        img = sample_imgs[i].numpy() * 0.5 + 0.5
        
        # Handle 1-channel (grayscale) vs 3-channel (RGB) formatting for matplotlib
        if img.shape[0] == 1:
            img = img.squeeze()
            cmap = 'gray'
        else:
            img = np.transpose(img, (1, 2, 0))  # Convert [C, H, W] to [H, W, C]
            cmap = None
            
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(full_train_dataset.classes[sample_labels[i].item()], fontsize=9)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "preprocessed_batch.png"), dpi=150)
    plt.close()

    return train_loader, val_loader, test_loader, full_train_dataset.classes