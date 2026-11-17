# 19.3 임베딩: 번호(정수)를 벡터(실수 여러 개)로. 학습되는 조회표.
import torch
import torch.nn as nn

torch.manual_seed(0)
vocab_size, dim = 10, 4
emb = nn.Embedding(vocab_size, dim)                   # 10개 토큰 각각에 4차원 벡터
print(emb.weight.shape)                               # [10, 4]: 조회표 자체가 파라미터

ids = torch.tensor([[3, 1, 0, 0],                     # 문장 2개, 길이 4 (0은 <pad>)
                    [5, 3, 9, 1]])
vec = emb(ids)
print("입력", tuple(ids.shape), "→ 출력", tuple(vec.shape))   # [2, 4] → [2, 4, 4] = [배치, 길이, 차원]
print("토큰 3 의 벡터:", [round(v, 3) for v in emb.weight[3].tolist()])
print("같은 토큰은 어디서나 같은 벡터:", torch.equal(vec[0, 0], vec[1, 1]))

# 문장 벡터 만들기 (가장 단순한 방법): 토큰 벡터의 평균. <pad> 는 빼고.
mask = (ids != 0).unsqueeze(-1)                       # [2, 4, 1] 패딩이 아닌 자리만 1
sent = (vec * mask).sum(1) / mask.sum(1)              # [2, 4]
print("문장 벡터 shape:", tuple(sent.shape))

# 이것이 곧 분류기: 임베딩 → 평균 → 선형층. RNN 도 Transformer 도 "평균" 자리를 더 좋은 것으로 바꾼 것이다.
clf = nn.Sequential(nn.Linear(dim, 2))
print("분류 점수:", tuple(clf(sent).shape))
