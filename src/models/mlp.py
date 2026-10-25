import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 10)
        )

    def features(self, x):
        return self.network[:-1](x)

    def forward(self, x):
        return self.network(x)

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.network(z).view(-1, 1, 28, 28)
