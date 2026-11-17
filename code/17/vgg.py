# 17.1 VGG: 같은 모양의 블록(3x3 합성곱 몇 개 + 풀링)을 반복해 깊게 쌓는다
import torch
import torch.nn as nn


def vgg_block(in_ch, out_ch, n_convs):
    """3x3 합성곱 n_convs 개 (크기 유지) + 2x2 풀링 (크기 절반)"""
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1),
                   nn.BatchNorm2d(out_ch),          # 각 채널의 값을 정규화 → 깊어져도 학습이 안정
                   nn.ReLU(inplace=True)]
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)                    # 층들을 순서대로 이어 붙인 하나의 모듈


# 블록 구성표: (출력 채널, 합성곱 수). 원 논문의 A(11층)·D(16층) 구성을 32px 입력에 맞게 줄인 것
CONFIGS = {
    "tiny":  [(16, 1), (32, 1), (64, 1)],                     # 32 → 16 → 8 → 4
    "small": [(32, 2), (64, 2), (128, 2)],
    "vgg11": [(64, 1), (128, 1), (256, 2), (512, 2), (512, 2)],   # 32 → 1
}


class VGG(nn.Module):
    def __init__(self, config="small", in_ch=3, num_classes=10, img_size=32):
        super().__init__()
        blocks, ch = [], in_ch
        for out_ch, n in CONFIGS[config]:
            blocks.append(vgg_block(ch, out_ch, n))
            ch = out_ch
        self.features = nn.Sequential(*blocks)
        side = img_size // (2 ** len(CONFIGS[config]))          # 블록 수만큼 절반
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(ch * side * side, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    for name in CONFIGS:
        m = VGG(name)
        n = sum(p.numel() for p in m.parameters())
        out = m(torch.zeros(2, 3, 32, 32))
        print(f"VGG-{name:<6} 블록 {len(CONFIGS[name])}  파라미터 {n:>10,}  [2,3,32,32] → {tuple(out.shape)}")
    print(VGG("tiny"))
