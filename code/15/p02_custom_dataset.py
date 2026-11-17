# 15.3~15.4 내 이미지 폴더를 Dataset 으로 만들고, transform 으로 전처리·증강한다
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "shapes")


class ImageFolderDataset(Dataset):
    """폴더 이름이 정답인 이미지 폴더. root/{클래스}/{파일}.png"""

    def __init__(self, root, transform=None):
        self.classes = sorted(os.listdir(root))                        # ['circle', 'square', 'triangle']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}  # 이름 → 번호
        self.items = []                                                # (경로, 번호) 목록
        for c in self.classes:
            folder = os.path.join(root, c)
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.items.append((os.path.join(folder, fname), self.class_to_idx[c]))
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")                        # 파일 → PIL 이미지 (여기서 비로소 읽는다)
        if self.transform:
            image = self.transform(image)                              # PIL → 텐서 (+ 전처리)
        return image, label


if __name__ == "__main__":          # import 될 때는 아래 실험이 돌지 않게 (4장)
    # ---- transform 없이: 목록만 만든다. 이미지는 아직 안 읽는다
    ds = ImageFolderDataset(os.path.join(ROOT, "train"))
    print("클래스:", ds.classes, ds.class_to_idx)
    print("len:", len(ds), "| items[0] =", ds.items[0][0].replace(ROOT, "…"), ds.items[0][1])
    img, label = ds[0]
    print("ds[0] ->", type(img).__name__, img.size, img.mode, "| label", label)

    # ---- 기본 transform: 텐서로 바꾸고 표준화
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    basic = transforms.Compose([
        transforms.Resize((32, 32)),           # 크기 통일
        transforms.ToTensor(),                 # [3, 32, 32], 0~1
        transforms.Normalize(mean, std),       # (x - mean) / std → 대략 -1~1
    ])
    ds = ImageFolderDataset(os.path.join(ROOT, "train"), transform=basic)
    img, label = ds[0]
    print("transform 후:", img.shape, img.dtype, f"범위 {img.min():.2f}~{img.max():.2f}", "| label", label)

    # ---- 증강(augmentation): 학습 때만, 매번 다르게
    augment = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds_aug = ImageFolderDataset(os.path.join(ROOT, "train"), transform=augment)
    torch.manual_seed(0)
    a, _ = ds_aug[0]
    b, _ = ds_aug[0]
    print("같은 이미지를 두 번 꺼내면 다른가:", not torch.equal(a, b))

    # ---- DataLoader 로 배치
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    images, labels = next(iter(loader))
    print("배치:", images.shape, labels.shape, "| labels[:10] =", labels[:10].tolist())

    # ---- 클래스별 개수 세기 (3장의 집계 그대로)
    counts = {}
    for _, label in ds.items:
        counts[ds.classes[label]] = counts.get(ds.classes[label], 0) + 1
    print("클래스별 개수:", counts)
