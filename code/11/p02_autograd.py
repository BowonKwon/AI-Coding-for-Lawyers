# 11.3 autograd: 기울기를 자동으로 계산한다
import torch

# 손실 L = (w * x - y)^2 에서 w 를 조금 바꾸면 L 이 얼마나 변하는가?
x = torch.tensor(2.0)
y = torch.tensor(7.0)
w = torch.tensor(1.0, requires_grad=True)      # "이 값에 대한 기울기를 추적하라"

pred = w * x                                    # 예측 2.0
loss = (pred - y) ** 2                          # (2 - 7)^2 = 25
print("예측", pred.item(), "손실", loss.item())

loss.backward()                                 # 기울기 계산 (역전파)
print("dL/dw =", w.grad.item())                 # 손으로 계산: 2*(w*x - y)*x = 2*(2-7)*2 = -20

# 경사 하강 한 걸음: 손실이 줄어드는 방향(기울기의 반대)으로 조금
lr = 0.1
with torch.no_grad():                           # 갱신 자체는 추적하지 않는다
    w -= lr * w.grad                            # 1.0 - 0.1*(-20) = 3.0
print("갱신된 w =", w.item())

# 기울기는 누적된다! 다음 걸음 전에 반드시 지운다 (10장의 ① zero_grad 가 하는 일)
w.grad.zero_()
pred = w * x
loss = (pred - y) ** 2
print("두 번째 걸음: 예측", pred.item(), "손실", loss.item())
loss.backward()
print("dL/dw =", w.grad.item())

# 여러 걸음 반복하면 w 가 정답(3.5)에 다가간다
w = torch.tensor(1.0, requires_grad=True)
for step in range(8):
    loss = (w * x - y) ** 2
    if w.grad is not None:
        w.grad.zero_()
    loss.backward()
    with torch.no_grad():
        w -= 0.05 * w.grad
    print(f"걸음 {step + 1}: w={w.item():.4f} 손실={loss.item():.4f}")
