# 13장 train.py: 12장 학습 코드에 (1) 명령줄 인자 (2) 결과 폴더와 설정 저장 (3) 가중치 저장 (4) 지표 기록을 더한 것.
# 실행 예:
#   python train.py                                  # 기본값으로
#   python train.py --epochs 3 --lr 0.05 --run lr005  # 인자를 바꿔서
#   python train.py --help                           # 인자 목록
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="MNIST MLP 학습")
    p.add_argument("--data-root", default=os.environ.get("JUCOS_DATA", "./data"), help="데이터 폴더")
    p.add_argument("--results", default="results", help="결과를 저장할 상위 폴더")
    p.add_argument("--run", default=None, help="이 실행의 이름 (없으면 시각으로)")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args()


class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x.flatten(1))))


def evaluate(model, loader, loss_fn, device):
    """정확도와 평균 손실을 함께 돌려준다."""
    model.eval()
    correct, total_loss = 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            total_loss += loss_fn(scores, labels).item() * len(labels)   # 배치 평균 × 개수 = 합
            correct += (scores.argmax(1) == labels).sum().item()
    model.train()
    n = len(loader.dataset)
    return correct / n, total_loss / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 결과 폴더: results/{run}/ 에 이 실행의 모든 것을 모은다
    run_name = args.run or time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.results, run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)          # 4장의 JSON: 설정을 그대로 남긴다

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = MLP(hidden=args.hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    history = []                                                         # 에포크별 지표를 모아 둔다
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        running, t0 = 0.0, time.time()
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % args.log_every == 0:
                print(f"  에포크 {epoch} 걸음 {step:4d}  평균 손실 {running / args.log_every:.4f}")
                running = 0.0
        acc, test_loss = evaluate(model, test_loader, loss_fn, device)
        history.append({"epoch": epoch, "test_acc": acc, "test_loss": test_loss, "seconds": time.time() - t0})
        print(f"에포크 {epoch}: 시험 정확도 {acc:.4f}  시험 손실 {test_loss:.4f}")
        if acc > best_acc:                                               # 가장 좋은 에포크의 가중치만 남긴다
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, "model_last.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_test_acc": best_acc, "history": history}, f, ensure_ascii=False, indent=2)
    print(f"저장 완료: {out_dir}/  (config.json, model_best.pt, model_last.pt, metrics.json)")


if __name__ == "__main__":
    main()
