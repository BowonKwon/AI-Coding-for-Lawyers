# 24장 같은 실험을 씨앗만 바꿔 여러 번: 점추정치에 불확실성을 붙인다.
# 14장 패키지의 train.py 를 함수로 불러 쓰지 않고, 여기서는 12장 크기의 작은 실험을 자체 정의한다.
import json
import math
import os
import statistics
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "19"))
from statutes_dataset import StatuteDataset

LAWS = ["민법", "형법", "민사소송법", "형사소송법"]
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


class BagOfWords(nn.Module):
    def __init__(self, vocab_size, dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, ids):
        vec = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1).float()
        return self.fc((vec * mask).sum(1) / mask.sum(1).clamp(min=1))


def one_run(seed, lr, epochs=6):
    torch.manual_seed(seed)
    train_set = StatuteDataset(LAWS, "train", seed=seed)                  # 분할도 씨앗에 따라 달라진다
    test_set = StatuteDataset(LAWS, "test", vocab=train_set.vocab, seed=seed)
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)
    model = BagOfWords(len(train_set.vocab), 64, len(LAWS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
    model.eval()
    correct, per_class = 0, {c: [0, 0] for c in LAWS}
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            for p, t in zip(pred.tolist(), y.tolist()):
                per_class[LAWS[t]][1] += 1
                per_class[LAWS[t]][0] += int(p == t)
    acc = correct / len(test_set)
    recall = {c: v[0] / v[1] for c, v in per_class.items()}                # 클래스별 재현율
    return {"seed": seed, "lr": lr, "acc": acc, "recall": recall, "n_test": len(test_set)}


def summarize(runs):
    accs = [r["acc"] for r in runs]
    m, s = statistics.mean(accs), statistics.stdev(accs) if len(accs) > 1 else 0.0
    ci = 1.96 * s / math.sqrt(len(accs))                                    # 정규 근사 95% 신뢰구간(씨앗 5개면 거칠다)
    return m, s, ci


if __name__ == "__main__":
    seeds = [0, 1, 2, 3, 4]
    all_runs = []
    for lr in (1e-3, 3e-3):
        runs = []
        t0 = time.time()
        for s in seeds:
            runs.append(one_run(s, lr))
        m, sd, ci = summarize(runs)
        print(f"lr={lr:<6} 정확도 {m:.3f} ± {ci:.3f} (95% CI, n={len(seeds)} 씨앗, SD {sd:.3f})  "
              f"개별 {[round(r['acc'], 3) for r in runs]}  ({time.time() - t0:.0f}초)")
        all_runs += runs
    # 클래스별 재현율(마지막 설정의 씨앗 평균): 정확도 하나로는 안 보이는 것
    last = [r for r in all_runs if r["lr"] == 3e-3]
    print("클래스별 재현율 (lr=3e-3, 씨앗 평균):",
          {c: round(statistics.mean(r["recall"][c] for r in last), 3) for c in LAWS})
    with open(os.path.join(OUT, "multiseed.json"), "w", encoding="utf-8") as f:
        json.dump(all_runs, f, ensure_ascii=False, indent=2)
    print("저장:", os.path.join(OUT, "multiseed.json"))
