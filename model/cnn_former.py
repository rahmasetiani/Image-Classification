import torch.nn as nn

class CNNFormer(nn.Module):
    def __init__(self, feature_dim, dff=1024, num_head=1, num_layer=1, n_class=2, dropout=0.1, device='cpu'):
        super(CNNFormer, self).__init__()
        self.layer = num_layer
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, 20, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout),
        ).to(device)

        self.hidden_dim = 20
        self.MHA = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_head, bias=False, dropout=dropout).to(
            device)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, self.hidden_dim)
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.lin_out = nn.Linear(self.hidden_dim * 256, n_class)

    def forward(self, x):
        x = self.conv(x)
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        for i in range(self.layer):
            y, _ = self.MHA(x, x, x)
            x = x + self.norm(y)
            y = self.feed_forward(x)
            x = x + self.norm(y)
        x = x.permute(0, 2, 1).view(batch_size, channels, height, width)
        x = x.reshape(batch_size, -1)
        x = self.lin_out(x)
        return x
