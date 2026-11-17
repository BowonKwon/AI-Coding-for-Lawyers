# 25.2 공개 언어 모델을 내 과제(법령 조문 4분류)에 맞추기: (A) 얼리고 분류층만 / (B) LoRA
# 준비: pip install transformers peft
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "19"))
from statutes_dataset import StatuteDataset                     # 원문·정답 목록만 빌려 쓴다 (토큰화는 HF 토크나이저로)

NAME = "monologg/koelectra-small-v3-discriminator"
LAWS = ["민법", "형법", "민사소송법", "형사소송법"]
tok = AutoTokenizer.from_pretrained(NAME)
torch.manual_seed(0)
raw_train = StatuteDataset(LAWS, "train")
raw_test = StatuteDataset(LAWS, "test", vocab=raw_train.vocab)


class HFDataset(Dataset):
    """조문 원문을 HF 토크나이저로 인코딩. 15·19장의 Dataset 뼈대 그대로."""

    def __init__(self, items, max_len=64):
        self.items, self.max_len = items, max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        text, label = self.items[i]
        enc = tok(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return enc["input_ids"][0], enc["attention_mask"][0], label


train_loader = DataLoader(HFDataset(raw_train.items), batch_size=32, shuffle=True)
test_loader = DataLoader(HFDataset(raw_test.items), batch_size=128)


def accuracy(model):
    model.eval()
    c = 0
    with torch.no_grad():
        for ids, mask, y in test_loader:
            c += (model(input_ids=ids, attention_mask=mask).logits.argmax(1) == y).sum().item()
    model.train()
    return c / len(raw_test)


def run(name, model, params, lr, epochs=int(os.environ.get("EPOCHS", 2))):   # EPOCHS 환경변수로 조절
    opt = torch.optim.AdamW(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    for ep in range(1, epochs + 1):
        for ids, mask, y in train_loader:
            opt.zero_grad()
            loss_fn(model(input_ids=ids, attention_mask=mask).logits, y).backward()
            opt.step()
        print(f"  {name} 에포크 {ep}: 시험 정확도 {accuracy(model):.3f}  ({time.time() - t0:.0f}초)")
    print(f"{name:<18} 학습 파라미터 {n_train:>10,} / 전체 {n_all:,} ({100 * n_train / n_all:.2f}%)")


if __name__ == "__main__":
    # A. 얼리고 분류층만 (18장의 특징 추출)
    torch.manual_seed(0)
    m = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=len(LAWS))
    for p in m.parameters():
        p.requires_grad = False
    for p in m.classifier.parameters():
        p.requires_grad = True
    run("A. 분류층만", m, list(m.classifier.parameters()), lr=1e-3)

    # B. LoRA: 어텐션의 query·value 선형층 옆에 작은 저계수 행렬을 붙여 그것만 학습
    torch.manual_seed(0)
    m = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=len(LAWS))
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1,
                     task_type="SEQ_CLS")
    m = get_peft_model(m, cfg)
    trainable = [p for p in m.parameters() if p.requires_grad]
    run("B. LoRA(r=8)", m, trainable, lr=5e-4)
    m.print_trainable_parameters()
