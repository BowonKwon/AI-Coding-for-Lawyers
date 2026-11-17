# 16.5 도형 데이터에서 MLP 와 LeNet: 위치가 바뀌는 입력에 합성곱이 왜 강한가
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "15"))
from p02_custom_dataset import ImageFolderDataset
from lenet import LeNet

ROOT = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "shapes")
tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
train_loader = DataLoader(ImageFolderDataset(os.path.join(ROOT, "train"), tf), batch_size=32, shuffle=True)
test_loader = DataLoader(ImageFolderDataset(os.path.join(ROOT, "test"), tf), batch_size=64)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(3 * 32 * 32, 128), nn.Linear(128, 3)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


def accuracy(model):
    model.eval()
    c = 0
    with torch.no_grad():
        for x, y in test_loader:
            c += (model(x).argmax(1) == y).sum().item()
    model.train()
    return c / len(test_loader.dataset)


def run(name, model, epochs=10, lr=0.01):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    accs = []
    for ep in range(1, epochs + 1):
        for x, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        accs.append(accuracy(model))
    n = sum(p.numel() for p in model.parameters())
    print(f"{name:<8} 파라미터 {n:>8,}  에포크별 정확도 {[round(a, 2) for a in accs]}  ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    torch.manual_seed(0); run("MLP", MLP())
    torch.manual_seed(0); run("LeNet", LeNet(in_ch=3, num_classes=3, img_size=32))
