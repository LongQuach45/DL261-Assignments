import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, in_channels=1, img_size=28, hidden_size=128, num_layers=2, num_classes=10, rnn_type="LSTM", dropout=0.2):
        super(RNNClassifier, self).__init__()
        input_size = in_channels * img_size
        rnn_cls = nn.LSTM if rnn_type.upper() == "LSTM" else nn.GRU

        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])