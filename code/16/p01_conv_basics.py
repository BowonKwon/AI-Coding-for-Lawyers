# 16.1 합성곱(convolution)이 하는 일을 작은 예로 본다
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ---- 5x5 흑백 이미지 하나: 가운데 세로선
img = torch.zeros(1, 1, 5, 5)                    # [배치, 채널, 높이, 너비]
img[0, 0, :, 2] = 1.0
print(img[0, 0])

# ---- 커널(필터) 3x3: "세로 경계"를 찾는 손으로 만든 필터
kernel = torch.tensor([[[[-1.0, 0.0, 1.0],
                         [-1.0, 0.0, 1.0],
                         [-1.0, 0.0, 1.0]]]])   # [출력채널 1, 입력채널 1, 3, 3]
out = F.conv2d(img, kernel)                      # 커널을 이미지 위로 밀며 곱해서 더한다
print("conv 출력 shape:", out.shape)             # [1, 1, 3, 3]: 5-3+1 = 3
print(out[0, 0])                                 # 선의 왼쪽은 +3, 오른쪽은 -3, 나머지 0

# ---- padding 으로 크기 유지, stride 로 건너뛰기
print("padding=1 :", F.conv2d(img, kernel, padding=1).shape)     # [1,1,5,5]
print("stride=2  :", F.conv2d(img, kernel, stride=2).shape)      # [1,1,2,2]

# ---- 풀링: 창 안의 최댓값만 남겨 크기를 반으로
x = torch.arange(16.0).reshape(1, 1, 4, 4)
print(x[0, 0])
print(F.max_pool2d(x, 2)[0, 0])                  # 2x2 창 → [2,2]

# ---- nn.Conv2d 층: 커널을 학습으로 찾는다. 파라미터 수 = 출력채널 × 입력채널 × k × k + 출력채널
conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
print(conv)
print("weight:", conv.weight.shape, "| bias:", conv.bias.shape,
      "| 파라미터", sum(p.numel() for p in conv.parameters()))   # 6*1*5*5 + 6 = 156
batch = torch.zeros(64, 1, 28, 28)
print("[64,1,28,28] →", conv(batch).shape)       # [64, 6, 24, 24]

# ---- 왜 MLP 보다 나은가: 같은 28x28 입력에 대해 첫 층의 파라미터 수 비교
fc = nn.Linear(784, 6 * 24 * 24)                 # MLP 가 같은 크기의 출력을 내려면
print("MLP 첫 층 파라미터:", sum(p.numel() for p in fc.parameters()), "vs Conv2d:", 156)
