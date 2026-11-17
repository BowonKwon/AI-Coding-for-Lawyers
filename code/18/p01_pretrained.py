# 18.1 남이 학습한 모델 불러오기: torchvision 의 ResNet-18 (ImageNet 사전학습)
import torch
from torchvision import models

weights = models.ResNet18_Weights.DEFAULT              # 사전학습 가중치 지정 (처음 한 번 내려받음, 약 45MB)
model = models.resnet18(weights=weights)
n = sum(p.numel() for p in model.parameters())
print(f"ResNet-18 파라미터 {n:,}")
print("마지막 층:", model.fc)                              # Linear(512 → 1000): ImageNet 의 1000 클래스

# 어떤 층들이 있나: 이름만 (17장의 stem / stages / head 와 대응)
for name, child in model.named_children():
    print(f"  {name:<8} {child.__class__.__name__}")

# 사전학습 모델이 요구하는 전처리 (크기·정규화) 도 가중치와 함께 온다
print("전처리:", weights.transforms())

# 가짜 이미지 하나를 넣어 본다
x = torch.zeros(1, 3, 224, 224)
model.eval()
with torch.no_grad():
    out = model(x)
print("출력:", tuple(out.shape), "| 상위 3 클래스 번호:", out[0].topk(3).indices.tolist())
print("클래스 이름 예:", [weights.meta["categories"][i] for i in out[0].topk(3).indices.tolist()])
