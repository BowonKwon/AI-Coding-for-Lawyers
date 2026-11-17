# 23장 문자 단위 언어 모델: "다음 글자는 무엇인가" 를 배우면 문장을 만들어 낼 수 있다.
# 데이터: 민법 조문 전체(약 19만 자). 모델: 22장의 Transformer 블록을 '앞만 보는' 마스크로 쓴 디코더.
import math
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "22"))
from transformer import PositionalEncoding, MultiHeadAttention

DATA = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "statutes", "민법.txt")


# ---- 1. 데이터: 글자 하나가 토큰. 어휘는 등장한 글자 전부
def load_text():
    with open(DATA, encoding="utf-8") as f:
        return f.read()


class CharData:
    def __init__(self, text, block=64, split=0.9):
        self.itos = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * split)
        self.train, self.val = ids[:n], ids[n:]
        self.block = block

    def encode(self, s):
        return torch.tensor([self.stoi.get(c, 0) for c in s], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def batch(self, split, B, g):
        """무작위 위치 B 개에서 block 길이의 (입력, 정답) 을 잘라 낸다. 정답 = 입력을 한 글자 밀어 놓은 것."""
        src = self.train if split == "train" else self.val
        ix = torch.randint(len(src) - self.block - 1, (B,), generator=g)
        x = torch.stack([src[i: i + self.block] for i in ix])
        y = torch.stack([src[i + 1: i + self.block + 1] for i in ix])
        return x, y


# ---- 2. 모델: 인코더 블록과 같되, 어텐션이 미래 글자를 보지 못하게 마스크
class CausalBlock(nn.Module):
    def __init__(self, d, h, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d, h)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        B, L, d = x.shape
        q = k = v = self.n1(x)
        # MultiHeadAttention 은 pad_mask 만 받으므로 여기서 직접 계산한다
        a = self.attn
        q = a.wq(q).view(B, L, a.h, a.dk).transpose(1, 2)
        k = a.wk(k).view(B, L, a.h, a.dk).transpose(1, 2)
        v = a.wv(v).view(B, L, a.h, a.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1) / math.sqrt(a.dk)).masked_fill(causal_mask[:L, :L], float("-inf"))
        out = (F.softmax(scores, -1) @ v).transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(a.wo(out))
        x = x + self.drop(self.ff(self.n2(x)))
        return x


class CharLM(nn.Module):
    def __init__(self, vocab, block=64, d=128, h=4, d_ff=256, n_layers=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = PositionalEncoding(d, block)
        self.blocks = nn.ModuleList([CausalBlock(d, h, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)                          # 각 위치에서 "다음 글자" 의 점수 vocab 개
        mask = torch.triu(torch.ones(block, block, dtype=torch.bool), diagonal=1)   # 위 삼각 = 미래
        self.register_buffer("mask", mask)
        self.block = block

    def forward(self, ids):                                       # [B, L] → [B, L, vocab]
        x = self.pos(self.emb(ids) * math.sqrt(self.emb.embedding_dim))
        for blk in self.blocks:
            x = blk(x, self.mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, ids, n_new, temperature=1.0, g=None):
        """앞 글자들을 주면 다음 글자를 뽑고, 그것을 다시 입력에 붙여 반복한다 (자기회귀)."""
        for _ in range(n_new):
            ctx = ids[:, -self.block:]
            logits = self(ctx)[:, -1] / temperature                # 마지막 위치의 점수만
            probs = F.softmax(logits, -1)
            nxt = torch.multinomial(probs, 1, generator=g)         # 확률대로 하나 뽑기
            ids = torch.cat([ids, nxt], 1)
        return ids


if __name__ == "__main__":
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    data = CharData(load_text(), block=64)
    print(f"글자 수 {len(data.train) + len(data.val):,} | 어휘(글자 종류) {len(data.itos)} | 학습 {len(data.train):,} 검증 {len(data.val):,}")
    x, y = data.batch("train", 2, g)
    print("입력:", repr(data.decode(x[0][:20].tolist())), "\n정답:", repr(data.decode(y[0][:20].tolist())), "(한 글자 밀림)")

    model = CharLM(len(data.itos))
    print(f"파라미터 {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    steps = int(os.environ.get("STEPS", 1500))
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = data.batch("train", 32, g)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))   # 모든 위치에서 다음 글자 맞히기
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = data.batch("val", 64, g)
                vl = F.cross_entropy(model(vx).reshape(-1, logits.size(-1)), vy.reshape(-1)).item()
            model.train()
            print(f"걸음 {step:5d}  학습 손실 {loss.item():.3f}  검증 손실 {vl:.3f}  (검증 perplexity {math.exp(vl):.1f})  {time.time() - t0:.0f}초")
    model.eval()
    for prompt in ["제1조(", "채무자가 ", "①"]:
        ids = data.encode(prompt).unsqueeze(0)
        out = model.generate(ids, 120, temperature=0.8, g=g)
        print(f"\n[{prompt}] → {data.decode(out[0].tolist())}")
