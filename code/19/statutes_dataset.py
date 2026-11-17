# 19.4 텍스트 Dataset: 조문 → (토큰 번호 텐서, 법령 번호). 15장의 ImageFolderDataset 과 같은 뼈대.
import os
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

DATA = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "statutes")
PAD, UNK = "<pad>", "<unk>"


def tokenize(s):
    return s.split()


class Vocab:
    def __init__(self, sentences, min_count=2, max_size=8000):
        counter = Counter(t for s in sentences for t in tokenize(s))
        self.itos = [PAD, UNK] + [t for t, c in counter.most_common(max_size) if c >= min_count]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s, max_len):
        ids = [self.stoi.get(t, 1) for t in tokenize(s)][:max_len]
        return ids + [0] * (max_len - len(ids))


class StatuteDataset(Dataset):
    """여러 법령의 조문을 모아 '어느 법령의 조문인가' 를 맞히는 분류 데이터셋."""

    def __init__(self, laws, split="train", vocab=None, max_len=40, test_ratio=0.2, seed=0):
        self.classes = list(laws)
        items = []
        for label, name in enumerate(laws):
            with open(os.path.join(DATA, f"{name}.txt"), encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            lines = [l for l in lines if len(tokenize(l)) >= 4]     # '제786조 삭제' 같은 빈 조문은 뺀다
            items += [(l, label) for l in lines]
        g = torch.Generator().manual_seed(seed)             # 같은 씨앗이면 같은 분할
        perm = torch.randperm(len(items), generator=g).tolist()
        n_test = int(len(items) * test_ratio)
        idx = perm[n_test:] if split == "train" else perm[:n_test]
        self.items = [items[i] for i in idx]
        self.vocab = vocab or Vocab([s for s, _ in self.items])   # 어휘는 학습 데이터로만 만든다
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        text, label = self.items[i]
        return torch.tensor(self.vocab.encode(text, self.max_len)), label


if __name__ == "__main__":
    laws = ["민법", "형법", "민사소송법", "형사소송법"]
    train = StatuteDataset(laws, "train")
    test = StatuteDataset(laws, "test", vocab=train.vocab)        # 시험 데이터는 학습 어휘를 그대로 쓴다
    print("학습", len(train), "시험", len(test), "| 어휘", len(train.vocab), "| 클래스", train.classes)
    x, y = train[0]
    print("train[0] ->", tuple(x.shape), x.dtype, "| label", y, train.classes[y])
    print("  원문:", train.items[0][0][:50])
    print("  복원:", " ".join(train.vocab.itos[i] for i in x.tolist() if i != 0)[:50])
    xb, yb = next(iter(DataLoader(train, batch_size=8, shuffle=True)))
    print("배치:", tuple(xb.shape), tuple(yb.shape))
    counts = Counter(train.classes[l] for _, l in train.items)
    print("학습 클래스별:", dict(counts))
