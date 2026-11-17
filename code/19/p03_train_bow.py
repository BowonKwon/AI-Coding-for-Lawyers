# 19.5 첫 텍스트 분류기: 임베딩 → 평균 → 선형층 ("단어 주머니" 모델). 12장 루프 그대로.
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from statutes_dataset import StatuteDataset

LAWS = ["민법", "형법", "민사소송법", "형사소송법"]
torch.manual_seed(0)
train_set = StatuteDataset(LAWS, "train")
test_set = StatuteDataset(LAWS, "test", vocab=train_set.vocab)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)


class BagOfWords(nn.Module):
    def __init__(self, vocab_size, dim, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)   # <pad> 의 벡터는 0 으로 고정
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, ids):                                       # ids: [B, L]
        vec = self.emb(ids)                                       # [B, L, dim]
        mask = (ids != 0).unsqueeze(-1).float()                   # 패딩 제외 평균
        sent = (vec * mask).sum(1) / mask.sum(1).clamp(min=1)     # [B, dim]
        return self.fc(sent)


model = BagOfWords(len(train_set.vocab), dim=64, num_classes=len(LAWS))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def accuracy():
    model.eval()
    c = 0
    with torch.no_grad():
        for x, y in test_loader:
            c += (model(x).argmax(1) == y).sum().item()
    model.train()
    return c / len(test_set)


print(f"학습 {len(train_set)} / 시험 {len(test_set)} / 어휘 {len(train_set.vocab)} / 파라미터 {sum(p.numel() for p in model.parameters()):,}")
print(f"학습 전 정확도: {accuracy():.3f}")
t0 = time.time()
for epoch in range(1, 11):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    if epoch % 2 == 0:
        print(f"에포크 {epoch:2d}: 손실 {loss.item():.4f}  시험 정확도 {accuracy():.3f}")
print(f"경과 {time.time() - t0:.0f}초")

# 어느 조문을 어느 법으로 보는가: 시험 데이터 몇 개
model.eval()
with torch.no_grad():
    for i in range(5):
        x, y = test_set[i]
        pred = model(x.unsqueeze(0)).argmax(1).item()
        mark = "" if pred == y else "  ← 틀림"
        print(f"  [{LAWS[y]} → 예측 {LAWS[pred]}]{mark} {test_set.items[i][0][:45]}…")
