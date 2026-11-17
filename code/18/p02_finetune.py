# 18.2 전이학습 세 방식 비교 (도형 데이터, ResNet-18)
#   A. 처음부터 학습 (사전학습 없음)
#   B. 특징 추출: 사전학습 가중치를 얼리고 마지막 층만 학습
#   C. 미세조정: 전부 학습하되 작은 학습률
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "15"))
from p02_custom_dataset import ImageFolderDataset

ROOT = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "shapes")
# 사전학습 모델의 전처리에 맞춘다: 크기 224 는 CPU 에서 느리므로 112 로 줄여 쓴다 (ResNet 은 크기에 유연)
tf = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   # ImageNet 통계
train_set = ImageFolderDataset(os.path.join(ROOT, "train"), tf)
test_set = ImageFolderDataset(os.path.join(ROOT, "test"), tf)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)
NUM_CLASSES = len(train_set.classes)


def accuracy(model):
    model.eval()
    c = 0
    with torch.no_grad():
        for x, y in test_loader:
            c += (model(x).argmax(1) == y).sum().item()
    model.train()
    return c / len(test_set)


def make_model(pretrained):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)      # 마지막 층을 내 문제(3 클래스)용으로 교체
    return m


def run(name, model, params, lr, epochs=2):
    torch.manual_seed(0)
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(epochs):
        for x, y in train_loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
    n_train = sum(p.numel() for p in params)
    print(f"{name:<22} 학습 파라미터 {n_train:>10,}  정확도 {accuracy(model):.3f}  ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    # A
    torch.manual_seed(0)
    m = make_model(pretrained=False)
    run("A. 처음부터", m, list(m.parameters()), lr=1e-3)
    # B
    torch.manual_seed(0)
    m = make_model(pretrained=True)
    for p in m.parameters():
        p.requires_grad = False                            # 얼리기
    for p in m.fc.parameters():
        p.requires_grad = True                             # 마지막 층만 학습
    run("B. 특징 추출(fc만)", m, list(m.fc.parameters()), lr=1e-3)
    # C
    torch.manual_seed(0)
    m = make_model(pretrained=True)
    run("C. 미세조정(전부)", m, list(m.parameters()), lr=1e-4)
