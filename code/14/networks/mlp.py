import torch
import torch.nn as nn


class MLP(nn.Module):
    """다층 퍼셉트론. 입력을 한 줄로 펴서 두 선형 층을 지난다."""

    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))
