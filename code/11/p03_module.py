# 11.4 nn.Module: 가중치를 가진 함수를 클래스로
import torch
import torch.nn as nn

torch.manual_seed(0)                                   # 난수 씨앗 고정 (결과 재현용)

# ---- 가장 작은 층: 선형 층 y = x @ W^T + b
linear = nn.Linear(3, 2)                               # 입력 3개 → 출력 2개
print(linear)
print("weight shape:", linear.weight.shape, "| bias shape:", linear.bias.shape)
x = torch.ones(4, 3)                                   # 배치 4개, 각 3차원
print("출력 shape:", linear(x).shape)                   # [4, 2]

# 가중치는 파라미터(학습 대상)이고 requires_grad=True 다
for name, p in linear.named_parameters():
    print(f"  {name}: {tuple(p.shape)}, requires_grad={p.requires_grad}")


# ---- 내 모델: 4장의 클래스·상속이 그대로
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()                             # 부모(nn.Module)의 준비를 먼저
        self.fc1 = nn.Linear(in_dim, hidden)           # 층을 속성으로 두면 자동으로 등록된다
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):                              # 계산 순서만 적는다
        x = self.fc1(x)
        x = torch.relu(x)                              # 비선형: 음수를 0으로
        x = self.fc2(x)
        return x


model = MLP(in_dim=784, hidden=128, out_dim=10)        # MNIST용: 784 픽셀 → 10개 숫자
print(model)

# 파라미터 수 세기
n_params = sum(p.numel() for p in model.parameters())
print("파라미터 수:", n_params)                         # 784*128+128 + 128*10+10 = 101,770

# 배치 하나를 통과시켜 본다
imgs = torch.rand(64, 1, 28, 28)                       # 가짜 이미지 64장
out = model(imgs.flatten(1))                           # [64, 784] → [64, 10]
print("출력 shape:", out.shape)
print("첫 이미지의 점수 10개:", [round(v, 2) for v in out[0].tolist()])
print("예측(가장 큰 점수의 자리):", out[0].argmax().item())

# state_dict: 가중치를 담은 딕셔너리 (3장) — 13장에서 저장할 것
sd = model.state_dict()
print("state_dict 키:", list(sd.keys()))
print("fc1.weight shape:", sd["fc1.weight"].shape)

# 흔한 실수: 모양이 안 맞으면 바로 오류
try:
    model(imgs)                                        # [64,1,28,28] 을 그대로 넣으면?
except RuntimeError as e:
    print("RuntimeError:", str(e)[:90], "...")
