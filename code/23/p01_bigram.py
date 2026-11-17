# 23.1 가장 작은 언어 모델: 바이그램. "이 글자 다음에 어떤 글자가 얼마나 자주 오는가" 를 세기만 한다.
import os
import math
import torch

DATA = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "statutes", "민법.txt")
text = open(DATA, encoding="utf-8").read()
itos = sorted(set(text))
stoi = {c: i for i, c in enumerate(itos)}
V = len(itos)
print(f"글자 {len(text):,} 개, 글자 종류 {V}")

# 세기: counts[a, b] = a 다음에 b 가 온 횟수
counts = torch.zeros(V, V)
ids = [stoi[c] for c in text]
for a, b in zip(ids, ids[1:]):
    counts[a, b] += 1

# 확률: 행마다 합이 1 이 되게. +1 은 한 번도 안 나온 쌍에 아주 작은 확률을 주기 위해(스무딩)
probs = (counts + 1) / (counts + 1).sum(1, keepdim=True)

# '채' 다음에 가장 잘 오는 글자
a = stoi["채"]
top = probs[a].topk(5)
print("'채' 다음:", [(itos[i], round(p, 3)) for p, i in zip(top.values.tolist(), top.indices.tolist())])

# 생성: 확률대로 다음 글자를 뽑아 이어 붙인다
g = torch.Generator().manual_seed(0)
cur = stoi["제"]
out = ["제"]
for _ in range(80):
    cur = torch.multinomial(probs[cur], 1, generator=g).item()
    out.append(itos[cur])
print("바이그램 생성:", "".join(out))

# 얼마나 놀랍지 않은가: 평균 -log p (교차 엔트로피) 와 perplexity
nll = -sum(math.log(probs[a, b].item()) for a, b in zip(ids[:20000], ids[1:20001])) / 20000
print(f"앞 2만 글자의 평균 손실 {nll:.3f}  perplexity {math.exp(nll):.1f}  (무작위 추측이면 {V})")
