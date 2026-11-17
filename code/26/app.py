# 26장 만든 모델 보여주기: 13장에서 학습한 MNIST 분류기를 그림판이 달린 웹 화면으로
# 준비: pip install gradio      실행: python app.py  → 브라우저에서 http://127.0.0.1:7860
import os
import sys
import json
import numpy as np
import torch
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "13"))
from train import MLP                                     # 13장의 모델 정의를 그대로

RUN_DIR = os.path.join(os.path.dirname(__file__), "..", "13", "results", "lr01")


def load_model():
    with open(os.path.join(RUN_DIR, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    m = MLP(hidden=cfg["hidden"])
    m.load_state_dict(torch.load(os.path.join(RUN_DIR, "model_best.pt"), map_location="cpu"))
    m.eval()
    return m


model = load_model()


def predict(image):
    """그림판 입력(딕셔너리 또는 배열) → 28x28 흑백 → 확률 10개"""
    if image is None:
        return {}
    if isinstance(image, dict):                            # Sketchpad 는 {"composite": 배열} 형태
        image = image["composite"]
    img = Image.fromarray(np.asarray(image)).convert("L").resize((28, 28))
    x = torch.tensor(np.asarray(img), dtype=torch.float32) / 255.0
    if x.mean() > 0.5:                                     # 흰 바탕에 검은 글씨면 반전 (MNIST 는 검은 바탕)
        x = 1.0 - x
    with torch.no_grad():
        probs = torch.softmax(model(x[None, None]), dim=1)[0]
    return {str(i): float(p) for i, p in enumerate(probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Sketchpad(label="숫자를 그려 보세요", image_mode="L"),
    outputs=gr.Label(num_top_classes=3, label="예측"),
    title="손글씨 숫자 분류기 (12·13장 MLP)",
    description="13장에서 학습한 모델(시험 정확도 94.7%)입니다. 그림판에 0~9를 그리면 확률을 보여 줍니다.",
)

if __name__ == "__main__":
    demo.launch()                                          # share=True 를 주면 1주일짜리 공개 링크가 생긴다
