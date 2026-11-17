# 12장 학습 파이프라인 한 바퀴: MNIST 손글씨 숫자를 MLP로 분류한다.
# 준비물 넷(데이터·모델·손실·옵티마이저) + 반복(걸음·에포크) + 평가.
# 실행: python train_mnist.py            (데이터는 DATA_ROOT 에 자동 다운로드, 약 64MB)

import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---- 0. 설정: 이 장에서는 딕셔너리 하나에 모아 둔다 (13장에서 argparse 로 바꾼다)
config = {
    "data_root": os.environ.get("JUCOS_DATA", "./data"),   # 데이터 폴더. 환경변수 JUCOS_DATA 가 있으면 그것, 없으면 ./data
    "batch_size": 64,
    "hidden": 128,
    "lr": 0.1,
    "epochs": 2,
    "seed": 0,
    "log_every": 200,           # 몇 걸음마다 손실을 찍을지
}
torch.manual_seed(config["seed"])
device = torch.device("cpu")    # Colab에서 GPU를 켰다면 torch.device("cuda")


# ---- 1. 데이터: (이미지, 정답) 쌍을 배치로 공급한다
transform = transforms.ToTensor()                              # PIL 이미지 → [1, 28, 28] float 텐서 (0~1)
train_set = datasets.MNIST(config["data_root"], train=True, download=True, transform=transform)
test_set = datasets.MNIST(config["data_root"], train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)   # 학습은 섞는다
test_loader = DataLoader(test_set, batch_size=1000)                                  # 평가는 순서 무관
print(f"학습 {len(train_set)}장 / 시험 {len(test_set)}장, 배치 {len(train_loader)}개")


# ---- 2. 모델: 11장의 MLP 그대로
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = x.flatten(1)                 # [B, 1, 28, 28] → [B, 784]  (모델 안에서 편다)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)               # 점수 10개 (logits)


model = MLP(784, config["hidden"], 10).to(device)

# ---- 3. 손실: 분류의 표준, 교차 엔트로피 (점수 → 확률 → 정답 확률의 -log)
loss_fn = nn.CrossEntropyLoss()

# ---- 4. 옵티마이저: 확률적 경사 하강법 (SGD). 파라미터 목록과 학습률을 준다
optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])


def evaluate(model, loader):
    """시험 데이터 전체에 대한 정확도. 학습하지 않으므로 기울기를 끈다."""
    model.eval()                                   # 평가 모드 (12장에서는 차이 없음, 습관)
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)                 # [B, 10]
            pred = scores.argmax(dim=1)            # 가장 큰 점수의 자리 = 예측 숫자
            correct += (pred == labels).sum().item()
    model.train()                                  # 학습 모드로 복귀
    return correct / len(loader.dataset)


# ---- 5. 반복: 걸음(배치) × 에포크
print(f"학습 전 정확도: {evaluate(model, test_loader):.3f}")
start = time.time()
for epoch in range(1, config["epochs"] + 1):
    running = 0.0
    for step, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                      # ① 기울기 초기화
        scores = model(images)                     # ② 예측
        loss = loss_fn(scores, labels)             # ③ 손실
        loss.backward()                            # ④ 기울기 계산
        optimizer.step()                           # ⑤ 가중치 갱신
        running += loss.item()
        if step % config["log_every"] == 0:
            print(f"  에포크 {epoch} 걸음 {step:4d}/{len(train_loader)}  평균 손실 {running / config['log_every']:.4f}")
            running = 0.0
    acc = evaluate(model, test_loader)
    print(f"에포크 {epoch} 끝: 시험 정확도 {acc:.3f}  (경과 {time.time() - start:.0f}초)")
