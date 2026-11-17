# 20.4 같은 데이터(법령 조문 4종 분류)에서 모델만 바꿔 비교: 단어 주머니 vs 직접 만든 RNN/LSTM vs nn.LSTM/GRU/BiLSTM
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "19"))
from statutes_dataset import StatuteDataset
from rnn_models import MyRNNClassifier, TorchRNNClassifier

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


def run(name, model, epochs=8, lr=2e-3):
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # 기울기 폭발 방지 (RNN 의 관례)
            opt.step()
        best = max(best, accuracy(model))
    n = sum(p.numel() for p in model.parameters())
    print(f"{name:<14} 파라미터 {n:>9,}  최고 정확도 {best:.3f}  마지막 {accuracy(model):.3f}  ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    print(f"학습 {len(train_set)} / 시험 {len(test_set)} / 어휘 {V}")
    run("단어 주머니", BagOfWords(V, 64, C))
    run("MyRNN", MyRNNClassifier(V, 64, 64, C, "rnn"))
    run("MyLSTM", MyRNNClassifier(V, 64, 64, C, "lstm"))
    run("nn.LSTM", TorchRNNClassifier(V, 64, 64, C, "lstm"))
    run("nn.GRU", TorchRNNClassifier(V, 64, 64, C, "gru"))
    run("BiLSTM", TorchRNNClassifier(V, 64, 64, C, "lstm", bidirectional=True))
