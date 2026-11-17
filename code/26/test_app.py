# 26장 검증: 브라우저 없이 app.py 의 predict 함수와 서버 기동을 확인한다 (6장의 '시험한다').
import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
import app

# 1) predict 가 시험 이미지에 대해 13장 infer.py 와 같은 답을 내는가
test_set = datasets.MNIST(os.environ.get("JUCOS_DATA", "./data"), train=False, download=True)
for i in (0, 1, 2, 8):
    img, label = test_set[i]                               # PIL, 검은 바탕 흰 글씨
    arr = 255 - np.asarray(img)                            # 그림판처럼 흰 바탕 검은 글씨로 뒤집어 넣는다
    probs = app.predict(arr)
    pred = max(probs, key=probs.get)
    print(f"시험 이미지 {i}: 정답 {label}  예측 {pred}  확신 {probs[pred]:.3f}")

# 2) 서버가 뜨고 HTTP 로 응답하는가
app.demo.launch(prevent_thread_lock=True, quiet=True)
import urllib.request
with urllib.request.urlopen("http://127.0.0.1:7860/", timeout=10) as r:
    html = r.read().decode("utf-8", "ignore")
print("서버 응답:", r.status, "| 제목 포함:", "손글씨" in html or "gradio" in html.lower())
app.demo.close()
print("서버 종료")
