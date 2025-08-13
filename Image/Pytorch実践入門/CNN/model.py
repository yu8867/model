import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_in_channels=3, n_out=10):
        super().__init__()
        self.conv1 = nn.Conv2d(n_in_channels, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, n_out)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))
        x = self.pool(self.act2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.act3(self.fc1(x))
        x = self.fc2(x)

        return x