# 11.1~11.2 텐서: 만들기, 모양(shape), 연산
import torch

# ---- 만들기
a = torch.tensor([1.0, 2.0, 3.0])                # 리스트에서
m = torch.tensor([[1.0, 2.0], [3.0, 4.0]])       # 2차원 (행렬)
z = torch.zeros(2, 3)                            # 0으로 채운 2행 3열
r = torch.rand(2, 3)                             # 0~1 난수
print(a)
print(m)
print(z)

# ---- 모양(shape)과 차원: 2부에서 가장 자주 보는 값
print("a.shape =", a.shape, "| 차원 수", a.dim())
print("m.shape =", m.shape)
img_batch = torch.zeros(64, 1, 28, 28)           # 8장에서 본 모양: 64장, 1채널, 28x28
print("img_batch.shape =", img_batch.shape, "| 원소 수", img_batch.numel())

# ---- 자료형과 장치
print(a.dtype, m.dtype, torch.tensor([1, 2]).dtype)   # float32가 기본, 정수 리스트는 int64
print(a.device)                                        # cpu
print("가속기 사용 가능:", torch.accelerator.is_available())

# ---- 연산: 같은 자리끼리(elementwise) vs 행렬곱
print(m + 1)                                     # 모든 원소에 1
print(m * m)                                     # 같은 자리끼리 곱
print(m @ m)                                     # 행렬곱 (2장 check_env 와 같음)
print(m.sum(), m.mean(), m.max())                # 집계 → 원소 하나짜리 텐서
print(m.sum().item())                            # .item(): 파이썬 숫자로 꺼내기

# ---- 모양 바꾸기: 원소는 그대로, 보는 틀만 바꾼다
x = torch.arange(12)                             # 0~11
print(x)
print(x.reshape(3, 4))
print(x.reshape(3, 4).T)                         # 전치
flat = img_batch.flatten(1)                      # 64장 각각을 한 줄로 펴기 (8장 코드의 torch.flatten(x, 1))
print("flatten(1) 후:", flat.shape)              # [64, 784]  (1*28*28 = 784)

# ---- 인덱싱과 슬라이싱: 3장의 리스트와 같다
print(m[0], m[0, 1], m[:, 1])                    # 첫 행, (0,1) 원소, 둘째 열
print(img_batch[0].shape)                        # 첫 번째 이미지 [1, 28, 28]

# ---- 브로드캐스팅: 모양이 달라도 규칙에 따라 맞춰 준다
v = torch.tensor([10.0, 20.0])
print(m + v)                                     # 각 행에 v를 더함
