import torch
import torch.nn as nn

class SparseAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.encoder2 = nn.Linear(256, 64)
        self.decoder = nn.Sequential(
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder1(x)
        return self.encoder2(h), h

    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x):
        z, h = self.encode(x)
        return self.decode(z), h 