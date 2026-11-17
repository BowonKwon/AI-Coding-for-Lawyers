# 15.1~15.2 Dataset 과 DataLoader 가 하는 일을 직접 본다
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

DATA_ROOT = os.environ.get("JUCOS_DATA", "./data")

# ---- torchvision 의 MNIST 도 결국 __len__ 과 __getitem__ 을 가진 Dataset 이다 (4장)
mnist = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transforms.ToTensor())
print("len:", len(mnist))
image, label = mnist[0]                          # __getitem__(0)
print("mnist[0] ->", type(image).__name__, image.shape, image.dtype, "| label", label, type(label).__name__)
print("픽셀 범위:", image.min().item(), "~", image.max().item())

# transform 을 빼면 무엇이 나오나
raw = datasets.MNIST(DATA_ROOT, train=True, download=True)          # transform 없음
img_raw, _ = raw[0]
print("transform 없이:", type(img_raw).__name__, img_raw.size, img_raw.mode)   # PIL 이미지 28x28, 흑백(L)


# ---- 내가 만든 Dataset: 리스트를 감싸는 가장 작은 예
class SquaresDataset(Dataset):
    """i번째 항목이 (i를 담은 텐서, i의 제곱) 인 데이터셋. 데이터가 파일이 아니어도 된다."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i < 0 or i >= self.n:
            raise IndexError(i)
        return torch.tensor([float(i)]), i * i


ds = SquaresDataset(10)
print("len:", len(ds), "| ds[3] ->", ds[3])

# ---- DataLoader: 묶고(batch), 섞고(shuffle), 하나씩 꺼내 준다
loader = DataLoader(ds, batch_size=4, shuffle=False)
for step, (x, y) in enumerate(loader):
    print(f"배치 {step}: x.shape={tuple(x.shape)} y={y.tolist()}")   # 마지막 배치는 2개

loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
torch.manual_seed(0)
for step, (x, y) in enumerate(loader):
    print(f"섞은 배치 {step}: y={y.tolist()}")                        # drop_last: 남는 2개는 버림

# ---- MNIST 를 배치로: 12장에서 매 걸음 받던 것이 바로 이것
loader = DataLoader(mnist, batch_size=64, shuffle=True)
images, labels = next(iter(loader))
print("MNIST 배치:", images.shape, labels.shape, "| labels[:8] =", labels[:8].tolist())
print("배치 수:", len(loader), "(=", len(mnist), "/ 64 올림)")
