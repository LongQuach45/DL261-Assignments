import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

def run_eda():

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
    plt.show()


def get_dataloaders(dataset_name="fashion_mnist", batch_size=64, seed=42):

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

    full_train_dataset = dataset_class(
        root='./data', train=True, download=True, transform=transform
    )

    train_size = 50000
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(seed)
    )

    test_dataset = dataset_class(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset: {dataset_name} | Split: Train={train_size}, Val={val_size}, Test={len(test_dataset)} | Seed={seed}[cite: 1]")
    return train_loader, val_loader, test_loader, full_train_dataset.classes