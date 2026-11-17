# 25.1 공개 언어 모델 불러오기: Hugging Face 의 한국어 소형 모델(KoELECTRA-small, 약 1,400만 파라미터)
# 준비: pip install transformers    (모델은 첫 실행 때 내려받아 ~/.cache/huggingface 에 둔다)
import torch
from transformers import AutoTokenizer, AutoModel

NAME = "monologg/koelectra-small-v3-discriminator"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME)
model.eval()

print(f"모델 {NAME}")
print(f"파라미터 {sum(p.numel() for p in model.parameters()):,} | 어휘 {tok.vocab_size:,}")

# ---- 토크나이저: 19장의 공백 단위와 달리 '서브워드' 로 쪼갠다. 모르는 단어가 거의 없다.
s = "제4조(성년) 사람은 19세로 성년에 이르게 된다."
enc = tok(s, return_tensors="pt")
print("\n원문  :", s)
print("토큰  :", tok.convert_ids_to_tokens(enc["input_ids"][0]))
print("번호  :", enc["input_ids"][0].tolist())
print("shape :", tuple(enc["input_ids"].shape), "(배치 1, 토큰 수)")

# ---- 모델 통과: 토큰마다 벡터(은닉 상태)가 나온다. 19장 임베딩 → 22장 인코더 블록을 지난 결과.
with torch.no_grad():
    out = model(**enc)
h = out.last_hidden_state
print("\n출력 shape:", tuple(h.shape), "= [배치, 토큰 수, 차원]")

# ---- 구조: 22장에서 만든 것과 같은 부품 이름이 보인다
for name, child in model.named_children():
    print(f"  {name:<11} {child.__class__.__name__}")
print("  encoder 층 수:", len(model.encoder.layer))
layer0 = model.encoder.layer[0]
for name, child in layer0.named_children():
    print(f"    layer[0].{name:<13} {child.__class__.__name__}")

# ---- 문장 벡터: [CLS] 토큰(첫 번째) 또는 평균. 두 조문의 유사도
def sent_vec(text):
    e = tok(text, return_tensors="pt")
    with torch.no_grad():
        return model(**e).last_hidden_state[0].mean(0)

pairs = [("사람은 19세로 성년에 이르게 된다.", "미성년자가 법률행위를 함에는 법정대리인의 동의를 얻어야 한다."),
         ("사람은 19세로 성년에 이르게 된다.", "검사는 공소를 제기하고 수행한다.")]
for a, b in pairs:
    va, vb = sent_vec(a), sent_vec(b)
    cos = torch.cosine_similarity(va, vb, dim=0).item()
    print(f"\n유사도 {cos:.3f} : {a[:18]}… ↔ {b[:18]}…")
