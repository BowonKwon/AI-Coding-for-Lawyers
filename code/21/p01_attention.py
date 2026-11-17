# 21장 어텐션의 계산을 가장 작은 예로: 질의(query) 하나가 키(key) 여러 개 중 어디를 얼마나 볼지
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# 문장 하나: 토큰 5개, 각 토큰은 4차원 벡터 (19장의 임베딩 결과라고 생각)
tokens = ["원고는", "피고에게", "손해를", "배상하라고", "청구하였다"]
x = torch.randn(5, 4)

# 1) 점수: 질의와 각 키의 내적. 여기서는 질의 = 키 = 값 = x (self-attention)
q = x[3]                                    # "배상하라고" 의 관점에서
scores = x @ q                              # [5]: 다른 토큰들과 얼마나 맞는가
print("점수      :", [round(s, 2) for s in scores.tolist()])

# 2) 가중치: softmax 로 합이 1 인 확률로
w = F.softmax(scores / x.shape[1] ** 0.5, dim=0)   # sqrt(d) 로 나누는 것은 값이 너무 커지는 것을 막기 위해
print("가중치    :", {t: round(v, 2) for t, v in zip(tokens, w.tolist())}, "| 합", round(w.sum().item(), 2))

# 3) 출력: 값(value) 들의 가중 평균
out = w @ x                                 # [4]
print("출력 shape:", tuple(out.shape), "= 가중치로 섞은 벡터")

# 모든 토큰이 동시에 질의가 되면 → 행렬 하나로 끝난다
W = F.softmax((x @ x.T) / x.shape[1] ** 0.5, dim=1)     # [5, 5]: i 행 = i 번 토큰이 각 토큰을 보는 가중치
print("\n어텐션 행렬 [5,5] (행 합 = 1):")
for t, row in zip(tokens, W.tolist()):
    print(f"  {t:<6}", [round(v, 2) for v in row])
print("출력:", tuple((W @ x).shape), "= 입력과 같은 [5, 4], 각 토큰이 문맥을 섞어 넣은 새 벡터")

# 19장의 '평균' 과 비교: 평균은 가중치가 전부 1/5 인 어텐션이다
print("\n단어 주머니(평균)의 가중치:", [round(1 / 5, 2)] * 5)
