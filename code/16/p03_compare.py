# 16.4 같은 데이터·같은 학습 루프에서 모델만 바꿔 비교: MLP vs LeNet vs LeNetWide vs LeNetDeep (MNIST, 1 에포크)
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from lenet import LeNet, LeNetWide, LeNetDeep

DATA_ROOT = os.environ.get("JUCOS_DATA", "./data")
tf = transforms.ToTensor()
train_set = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tf)
test_set = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tf)
test_loader = DataLoader(test_set, batch_size=1000)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(784, 128), nn.Linear(128, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


def accuracy(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
    model.train()
    return correct / len(test_set)


def run(cls, epochs=1, lr=0.05):
    torch.manual_seed(0)
    model = cls()
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    t0 = time.time()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    n = sum(p.numel() for p in model.parameters())
    print(f"{cls.__name__:<10} 파라미터 {n:>8,}  정확도 {accuracy(model):.4f}  ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    for cls in (MLP, LeNet, LeNetWide, LeNetDeep):
        run(cls)
