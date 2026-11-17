# 15.5 데이터만 바꿔 끼우기: 12장의 학습 루프에 도형 데이터셋을 넣는다. 모델은 MLP 그대로(입력 크기만 다름).
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from p02_custom_dataset import ImageFolderDataset

ROOT = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "shapes")
torch.manual_seed(0)

tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),
                         transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
train_set = ImageFolderDataset(os.path.join(ROOT, "train"), transform=tf)
test_set = ImageFolderDataset(os.path.join(ROOT, "test"), transform=tf)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


model = MLP(3 * 32 * 32, 128, len(train_set.classes))     # 입력 3072, 출력 3 — 바뀐 것은 이 숫자뿐
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)


def accuracy(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
    model.train()
    return correct / len(loader.dataset)


print(f"학습 {len(train_set)}장 / 시험 {len(test_set)}장 / 클래스 {train_set.classes}")
print(f"학습 전 정확도: {accuracy(test_loader):.3f}")
t0 = time.time()
for epoch in range(1, 11):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        print(f"에포크 {epoch:2d}: 손실 {loss.item():.4f}  시험 정확도 {accuracy(test_loader):.3f}")
print(f"경과 {time.time() - t0:.0f}초")
