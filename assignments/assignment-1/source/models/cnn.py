import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, in_channels=1, img_size=28, base_channels=16, fc_dim=128, num_classes=10):
        super(CNNClassifier, self).__init__()
        feat_size = img_size // 4

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear((base_channels * 2) * feat_size * feat_size, fc_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)