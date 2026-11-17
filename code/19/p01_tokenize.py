# 19.1~19.2 텍스트 → 숫자: 토큰화, 어휘 사전, 패딩
import os
from collections import Counter
import torch

DATA = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "statutes")


def load_lines(name):
    with open(os.path.join(DATA, f"{name}.txt"), encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


civil = load_lines("민법")
crim = load_lines("형법")
print("민법 조문", len(civil), "개 | 형법 조문", len(crim), "개")
print("예:", civil[0][:60])

# ---- 1) 토큰화: 문장을 조각(토큰)으로. 가장 단순한 방법 둘 — 글자 단위, 공백 단위
s = civil[3]
print("\n원문     :", s)
print("글자 토큰:", list(s)[:20], "… 총", len(s))
print("공백 토큰:", s.split()[:10], "… 총", len(s.split()))

# ---- 2) 어휘 사전: 토큰 → 번호. 자주 나오는 순으로 번호를 준다
PAD, UNK = "<pad>", "<unk>"                     # 특수 토큰: 빈칸 채우기, 모르는 토큰


def build_vocab(sentences, tokenizer, min_count=2, max_size=5000):
    counter = Counter(tok for s in sentences for tok in tokenizer(s))
    tokens = [t for t, c in counter.most_common(max_size) if c >= min_count]
    itos = [PAD, UNK] + tokens                  # index → token
    stoi = {t: i for i, t in enumerate(itos)}   # token → index (3장의 딕셔너리)
    return itos, stoi


tokenize = lambda s: s.split()
itos, stoi = build_vocab(civil + crim, tokenize)
print("\n어휘 크기:", len(itos), "| 상위 10:", itos[2:12])
print("'권리' →", stoi.get("권리"), "| '없는말' →", stoi.get("없는말", stoi[UNK]))


def encode(s, stoi, tokenizer, max_len):
    ids = [stoi.get(t, stoi[UNK]) for t in tokenizer(s)][:max_len]     # 자르고
    return ids + [stoi[PAD]] * (max_len - len(ids))                     # 채운다 (패딩)


# ---- 3) 길이 맞추기: 문장마다 길이가 다르므로 max_len 으로 자르거나 <pad> 로 채운다
for s in (civil[3], civil[0]):
    ids = encode(s, stoi, tokenize, max_len=12)
    print(f"\n{s[:40]}…")
    print("  ids :", ids)
    print("  복원:", [itos[i] for i in ids])

# ---- 4) 배치: 이제 텐서로 쌓을 수 있다 (15장의 DataLoader 가 하던 일)
batch = torch.tensor([encode(s, stoi, tokenize, 12) for s in civil[:4]])
print("\n배치 shape:", batch.shape, "dtype:", batch.dtype)          # [4, 12] int64

# ---- 5) 조문 길이 분포: max_len 을 얼마로 잡을지 데이터가 말해 준다
lens = sorted(len(tokenize(s)) for s in civil + crim)
print(f"\n토큰 길이: 최소 {lens[0]}  중앙값 {lens[len(lens)//2]}  90% {lens[int(len(lens)*0.9)]}  최대 {lens[-1]}")
