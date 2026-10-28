import torch.nn as nn


class CustomFoodCNN(nn.Module):
    """
    Self-implemented 4-block CNN baseline for 3x224x224 fine-grained food classification.
    Contains Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Dropout, and a 10-class linear head.
    Total trainable parameters: 1,242,090.
    """
    def __init__(self, in_channels=3, num_classes=10, base_channels=32, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=False),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def get_gradcam_target_layer(self):
        return self.features[-2]  # Final ReLU before Block 4 MaxPool

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)