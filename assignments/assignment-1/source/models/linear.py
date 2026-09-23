import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear(x)
        return logits