import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, in_channels=1, img_size=28, hidden_dim=256, num_classes=10):
        super(MLPClassifier, self).__init__()
        input_dim = in_channels * img_size * img_size
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.network(self.flatten(x))