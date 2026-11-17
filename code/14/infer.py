# 14장 infer.py: 패키지 구조에서의 추론. 모델 이름·설정은 config.json 에서 읽는다.
import os

import torch
from torchvision import datasets, transforms

from networks import build_model
from utils import parse_infer_args, load_json


def load_model(out_dir, which):
    config = load_json(os.path.join(out_dir, "config.json"))
    model = build_model(config["model"], hidden=config["hidden"])
    model.load_state_dict(torch.load(os.path.join(out_dir, f"model_{which}.pt"), map_location="cpu"))
    model.eval()
    return model, config


def main():
    args = parse_infer_args()
    out_dir = os.path.join(args.results, args.run)
    model, config = load_model(out_dir, args.which)
    print(f"모델 불러옴: {out_dir}/model_{args.which}.pt  ({config['model']}, hidden={config['hidden']})")
    test_set = datasets.MNIST(config["data_root"], train=False, download=True, transform=transforms.ToTensor())
    indices = [args.index] if args.index is not None else list(range(8))
    with torch.no_grad():
        for i in indices:
            image, label = test_set[i]
            probs = torch.softmax(model(image.unsqueeze(0)), dim=1)[0]
            pred = probs.argmax().item()
            print(f"시험 이미지 {i:5d}: 정답 {label}  예측 {pred}  확신 {probs[pred]:.3f}{'' if pred == label else '  ← 틀림'}")


if __name__ == "__main__":
    main()
