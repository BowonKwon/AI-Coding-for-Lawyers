# 12장 실험: 준비물 하나씩 바꿔 보기. train_mnist.py 를 함수로 감싼 판.
# 실행: python experiments.py
import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_ROOT = os.environ.get("JUCOS_DATA", "./data")
torch.manual_seed(0)
transform = transforms.ToTensor()
train_set = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
test_set = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=1000)


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


def accuracy(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(1) == labels).sum().item()
    model.train()
    return correct / len(test_set)


def run(name, lr=0.1, opt="sgd", zero_grad=True, batch_size=64, max_steps=None, seed=0):
    """준비물 하나를 바꿔 1 에포크 학습하고 정확도를 돌려준다."""
    torch.manual_seed(seed)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr) if opt == "sgd" else torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    last_losses = []
    for step, (images, labels) in enumerate(loader, start=1):
        if zero_grad:
            optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        last_losses.append(loss.item())
        if max_steps and step >= max_steps:
            break
    tail = sum(last_losses[-50:]) / min(50, len(last_losses))
    acc = accuracy(model)
    print(f"{name:<28} 마지막 50걸음 평균 손실 {tail:8.4f}  정확도 {acc:.3f}  ({time.time() - t0:.0f}초)")
    return acc


if __name__ == "__main__":
    run("기준 (SGD, lr=0.1)")
    run("학습률 너무 작음 (lr=0.001)", lr=0.001)
    run("학습률 너무 큼 (lr=10)", lr=10)
    run("zero_grad 빠뜨림", zero_grad=False)
    run("옵티마이저 Adam (lr=0.001)", lr=0.001, opt="adam")
    run("배치 1 (300걸음만)", batch_size=1, max_steps=300)
    run("배치 60000 (한 걸음)", batch_size=60000)
