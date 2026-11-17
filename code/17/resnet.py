# 17.2 ResNet: 블록의 입력을 출력에 더하는 "지름길(skip connection)" 로 아주 깊게 쌓아도 학습되게 한다
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """3x3 합성곱 둘 + 지름길. out = relu( F(x) + x )"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # 지름길: 모양(채널·크기)이 바뀌는 블록에서는 1x1 합성곱으로 x 의 모양을 맞춘다
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))        # 더하기 한 줄이 ResNet 의 전부


class ResNet(nn.Module):
    """입력부 → 단계(stage) 몇 개 → 출력부. 단계마다 채널은 두 배, 크기는 절반."""

    def __init__(self, blocks_per_stage=(2, 2, 2), channels=(16, 32, 64), in_ch=3, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, channels[0], 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        stages, ch = [], channels[0]
        for i, (n, out_ch) in enumerate(zip(blocks_per_stage, channels)):
            stride = 1 if i == 0 else 2                       # 첫 단계 외에는 크기 절반
            layers = [BasicBlock(ch, out_ch, stride)] + [BasicBlock(out_ch, out_ch) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            ch = out_ch
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(ch, num_classes))

    def forward(self, x):
        return self.head(self.stages(self.stem(x)))


if __name__ == "__main__":
    for cfg in [(1, 1, 1), (2, 2, 2), (3, 3, 3)]:
        m = ResNet(cfg)
        n = sum(p.numel() for p in m.parameters())
        depth = 1 + 2 * sum(cfg) + 1
        print(f"ResNet 블록 {cfg}  합성곱 층 수 약 {depth:>2}  파라미터 {n:>9,}  [2,3,32,32] → {tuple(m(torch.zeros(2, 3, 32, 32)).shape)}")
    # 지름길이 모양을 맞추는지
    b = BasicBlock(16, 32, stride=2)
    print("BasicBlock(16→32, stride 2): [2,16,32,32] →", tuple(b(torch.zeros(2, 16, 32, 32)).shape))
