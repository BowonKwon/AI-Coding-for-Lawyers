# 17.3 깊이 실험: 지름길이 없으면 깊은 망이 왜 안 배우는지 본다 (도형 데이터, 짧은 학습)
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "15"))
from p02_custom_dataset import ImageFolderDataset       # 15장의 Dataset 재사용

from vgg import VGG
from resnet import ResNet, BasicBlock

ROOT = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "shapes")
tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
train_loader = DataLoader(ImageFolderDataset(os.path.join(ROOT, "train"), tf), batch_size=32, shuffle=True)
test_loader = DataLoader(ImageFolderDataset(os.path.join(ROOT, "test"), tf), batch_size=64)


class PlainDeep(nn.Module):
    """지름길 없는 깊은 망: 3x3 합성곱 n 개를 그냥 쌓는다 (BatchNorm 도 없이)"""

    def __init__(self, n_layers, ch=16, num_classes=3):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(ch, num_classes))

    def forward(self, x):
        return self.head(self.body(x))


def accuracy(model):
    model.eval()
    c = 0
    with torch.no_grad():
        for x, y in test_loader:
            c += (model(x).argmax(1) == y).sum().item()
    model.train()
    return c / len(test_loader.dataset)


def run(name, model, epochs=6, lr=0.05):
    torch.manual_seed(0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    n = sum(p.numel() for p in model.parameters())
    print(f"{name:<26} 파라미터 {n:>9,}  정확도 {accuracy(model):.3f}  ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    torch.manual_seed(0); run("Plain 4층", PlainDeep(4))
    torch.manual_seed(0); run("Plain 20층 (지름길 없음)", PlainDeep(20))
    torch.manual_seed(0); run("VGG-tiny (BN 있음)", VGG("tiny", num_classes=3))
    torch.manual_seed(0); run("ResNet (1,1,1)", ResNet((1, 1, 1), num_classes=3))
    torch.manual_seed(0); run("ResNet (3,3,3) 약 20층", ResNet((3, 3, 3), num_classes=3))
