# 13장 infer.py: 저장된 가중치를 불러와 새 입력에 대해 예측한다. 학습 코드는 없다.
# 실행 예:
#   python infer.py --run lr005                # results/lr005/ 의 모델로 시험 이미지 8장 예측
#   python infer.py --run lr005 --index 42     # 42번째 시험 이미지 하나
import argparse
import json
import os

import torch
from torchvision import datasets, transforms

from train import MLP                        # 같은 모델 정의를 다시 쓴다 (복사하지 않는다)


def parse_args():
    p = argparse.ArgumentParser(description="MNIST MLP 추론")
    p.add_argument("--results", default="results")
    p.add_argument("--run", required=True, help="불러올 실행 이름 (results/{run}/)")
    p.add_argument("--which", default="best", choices=["best", "last"])
    p.add_argument("--index", type=int, default=None, help="시험 이미지 하나의 번호. 없으면 앞 8장")
    return p.parse_args()


def load_model(out_dir, which):
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)                                            # 학습 때의 설정을 읽어
    model = MLP(hidden=config["hidden"])                                 # 같은 모양의 모델을 만들고
    state = torch.load(os.path.join(out_dir, f"model_{which}.pt"), map_location="cpu")
    model.load_state_dict(state)                                         # 가중치를 채운다
    model.eval()                                                         # 추론 모드
    return model, config


def main():
    args = parse_args()
    out_dir = os.path.join(args.results, args.run)
    model, config = load_model(out_dir, args.which)
    print(f"모델 불러옴: {out_dir}/model_{args.which}.pt  (hidden={config['hidden']}, 학습 lr={config['lr']})")

    test_set = datasets.MNIST(config["data_root"], train=False, download=True, transform=transforms.ToTensor())
    indices = [args.index] if args.index is not None else list(range(8))

    with torch.no_grad():                                                # 추론에는 기울기가 필요 없다
        for i in indices:
            image, label = test_set[i]                                   # 4장의 __getitem__: (텐서, 정답)
            scores = model(image.unsqueeze(0))                           # 배치 차원 추가 [1,1,28,28]
            probs = torch.softmax(scores, dim=1)[0]                      # 점수 → 확률
            pred = probs.argmax().item()
            mark = "" if pred == label else "  ← 틀림"
            print(f"시험 이미지 {i:5d}: 정답 {label}  예측 {pred}  확신 {probs[pred]:.3f}{mark}")


if __name__ == "__main__":
    main()
