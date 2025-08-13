import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_in=3072, n_out=10):
        super().__init__()
        self.fc1 = nn.Linear(n_in, 512)
        self.fc2 = nn.Linear(512, n_out)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x