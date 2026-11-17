# 22.5 같은 데이터에서 단어 주머니 · BiLSTM · Transformer 비교, 그리고 어텐션이 무엇을 보는지
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "19"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "20"))
from statutes_dataset import StatuteDataset
from rnn_models import TorchRNNClassifier
from transformer import TransformerClassifier

LAWS = ["민법", "형법", "민사소송법", "형사소송법"]
torch.manual_seed(0)
train_set = StatuteDataset(LAWS, "train")
test_set = StatuteDataset(LAWS, "test", vocab=train_set.vocab)
test_loader = DataLoader(test_set, batch_size=256)
V, C = len(train_set.vocab), len(LAWS)


class BagOfWords(nn.Module):
    def __init__(self, vocab_size, dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, ids):
        vec = self.emb(ids)
        mask = (ids != 0).unsqueeze(-1).float()
        return self.fc((vec * mask).sum(1) / mask.sum(1).clamp(min=1))


def accuracy(model):
    model.eval()
    c = 0
    with torch.no_grad():
        for x, y in test_loader:
            c += (model(x).argmax(1) == y).sum().item()
    model.train()
    return c / len(test_set)


def run(name, model, epochs=8, lr=1e-3):
    torch.manual_seed(0)
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()
    best = 0.0
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        best = max(best, accuracy(model))
    n = sum(p.numel() for p in model.parameters())
    print(f"{name:<12} 파라미터 {n:>9,}  최고 정확도 {best:.3f}  마지막 {accuracy(model):.3f}  ({time.time() - t0:.0f}초)")
    return model


if __name__ == "__main__":
    print(f"학습 {len(train_set)} / 시험 {len(test_set)} / 어휘 {V}")
    run("단어 주머니", BagOfWords(V, 64, C), lr=2e-3)
    run("BiLSTM", TorchRNNClassifier(V, 64, 64, C, "lstm", bidirectional=True), lr=2e-3)
    tf = run("Transformer", TransformerClassifier(V, C, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=64))

    # 학습된 Transformer 가 어느 토큰을 보는가: 시험 조문 하나의 마지막 층 어텐션(헤드 평균), 각 토큰이 가장 많이 본 토큰
    tf.eval()
    x, y = test_set[3]
    text = test_set.items[3][0]
    toks = [train_set.vocab.itos[i] for i in x.tolist() if i != 0]
    with torch.no_grad():
        out, attns = tf(x.unsqueeze(0), return_attn=True)
    a = attns[-1][0].mean(0)[: len(toks), : len(toks)]                     # [L, L]
    print(f"\n조문: {text[:60]}…  (정답 {LAWS[y]}, 예측 {LAWS[out.argmax().item()]})")
    print("각 토큰이 가장 크게 본 토큰 (마지막 층, 헤드 평균):")
    for i, t in enumerate(toks[:10]):
        j = a[i].argmax().item()
        print(f"  {t:<12} → {toks[j]:<12} ({a[i, j]:.2f})")
