import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_channels * img_size * img_size, num_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))