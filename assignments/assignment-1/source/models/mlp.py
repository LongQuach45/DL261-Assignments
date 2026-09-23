import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=256, num_classes=10):
        super(MLPClassifier, self).__init__()
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
        x = self.flatten(x)
        logits = self.network(x)
        return logits