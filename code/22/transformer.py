# 22장 Transformer 인코더를 직접 만든다: 위치 인코딩, 멀티헤드 어텐션, 인코더 블록, 분류기
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """토큰 순서 정보를 더해 준다. 어텐션 자체는 순서를 모르기 때문."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()                    # [L, 1]
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)                                  # 짝수 차원: sin
        pe[:, 1::2] = torch.cos(pos * div)                                  # 홀수 차원: cos
        self.register_buffer("pe", pe)                                      # 파라미터 아님(학습 안 함), 저장은 됨

    def forward(self, x):                                                   # x: [B, L, d]
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    """헤드 h 개가 각자 다른 관점으로 어텐션을 계산하고 결과를 이어 붙인다."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)                                # 질의·키·값을 만드는 선형층
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)                                # 헤드를 합친 뒤 한 번 더

    def forward(self, x, pad_mask=None):                                    # x: [B, L, d], pad_mask: [B, L] True=패딩
        B, L, d = x.shape
        # [B, L, d] → [B, h, L, dk] : 헤드마다 dk 차원씩 나눠 본다
        q = self.wq(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.wk(x).view(B, L, self.h, self.dk).transpose(1, 2)
        v = self.wv(x).view(B, L, self.h, self.dk).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)               # [B, h, L, L]  (21장의 x @ x.T)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask[:, None, None, :], float("-inf"))   # 패딩은 보지 않는다
        attn = F.softmax(scores, dim=-1)                                    # 행 합 = 1
        out = attn @ v                                                      # [B, h, L, dk]
        out = out.transpose(1, 2).reshape(B, L, d)                          # 헤드를 다시 이어 붙임
        return self.wo(out), attn


class EncoderBlock(nn.Module):
    """어텐션 → 잔차+정규화 → 피드포워드 → 잔차+정규화. 17장 ResNet 의 지름길이 여기에도 있다."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        a, attn = self.attn(self.norm1(x), pad_mask)
        x = x + self.drop(a)                                                # 잔차 연결 1
        x = x + self.drop(self.ff(self.norm2(x)))                           # 잔차 연결 2
        return x, attn


class TransformerClassifier(nn.Module):
    """임베딩 + 위치 → 인코더 블록 n 개 → 패딩 제외 평균 → 분류"""

    def __init__(self, vocab_size, num_classes, d_model=64, n_heads=4, d_ff=128, n_layers=2, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, ids, return_attn=False):                              # ids: [B, L]
        pad_mask = ids == 0
        x = self.pos(self.emb(ids) * math.sqrt(self.emb.embedding_dim))
        attns = []
        for blk in self.blocks:
            x, attn = blk(x, pad_mask)
            attns.append(attn)
        x = self.norm(x)
        keep = (~pad_mask).unsqueeze(-1).float()
        sent = (x * keep).sum(1) / keep.sum(1).clamp(min=1)                 # 19장과 같은 평균 — 다만 문맥이 섞인 뒤
        out = self.fc(sent)
        return (out, attns) if return_attn else out


if __name__ == "__main__":
    torch.manual_seed(0)
    ids = torch.tensor([[5, 3, 9, 0, 0], [7, 2, 4, 8, 1]])
    m = TransformerClassifier(vocab_size=20, num_classes=3)
    out, attns = m(ids, return_attn=True)
    print("파라미터:", f"{sum(p.numel() for p in m.parameters()):,}")
    print("출력:", tuple(out.shape), "| 블록 수", len(attns), "| 어텐션 shape [B, h, L, L]:", tuple(attns[0].shape))
    print("첫 문장 1층 헤드0 어텐션 (패딩 열은 0):")
    for row in attns[0][0, 0].tolist():
        print("  ", [round(v, 2) for v in row])
    pe = PositionalEncoding(8, 6).pe
    print("위치 인코딩 [6, 8] 앞 3행:")
    for row in pe[:3].tolist():
        print("  ", [round(v, 2) for v in row])
