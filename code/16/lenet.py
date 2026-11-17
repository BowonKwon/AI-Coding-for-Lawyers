# 16.2 LeNet 계열 CNN: 합성곱 두 번 → 펴기 → 선형 두 번. 14장 networks 패키지에 넣을 수 있는 형태.
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """입력 [B, in_ch, 28, 28] (MNIST) 또는 [B, in_ch, 32, 32] (도형·CIFAR) 를 받는다."""

    def __init__(self, in_ch=1, num_classes=10, img_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 6, kernel_size=5)         # 28 → 24 (32 → 28)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)            # 12 → 8  (14 → 10)  (풀링 뒤)
        side = ((img_size - 4) // 2 - 4) // 2                     # 풀링 두 번 뒤의 한 변: 28→4, 32→5
        self.fc1 = nn.Linear(16 * side * side, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 합성곱 → 활성화 → 풀링
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)                             # [B, 16*side*side]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNetWide(LeNet):
    """구조 변경 실험 1: 채널을 두 배로. 부모의 forward 는 그대로 쓴다."""

    def __init__(self, in_ch=1, num_classes=10, img_size=28):
        super().__init__(in_ch, num_classes, img_size)
        side = ((img_size - 4) // 2 - 4) // 2
        self.conv1 = nn.Conv2d(in_ch, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * side * side, 120)


class LeNetDeep(nn.Module):
    """구조 변경 실험 2: 합성곱을 한 층 더, padding 으로 크기를 유지하며 쌓기."""

    def __init__(self, in_ch=1, num_classes=10, img_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, padding=1)          # 크기 유지
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        side = img_size // 8                                      # 풀링 세 번: 28→3, 32→4
        self.fc = nn.Linear(64 * side * side, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return self.fc(x.flatten(1))


if __name__ == "__main__":
    for cls in (LeNet, LeNetWide, LeNetDeep):
        m = cls()
        n = sum(p.numel() for p in m.parameters())
        print(f"{cls.__name__:<10} 파라미터 {n:>8,}  [64,1,28,28] → {tuple(m(torch.zeros(64, 1, 28, 28)).shape)}")
    m = LeNet(in_ch=3, num_classes=3, img_size=32)
    print(f"LeNet(3ch,32px) 파라미터 {sum(p.numel() for p in m.parameters()):,}  [8,3,32,32] → {tuple(m(torch.zeros(8, 3, 32, 32)).shape)}")
