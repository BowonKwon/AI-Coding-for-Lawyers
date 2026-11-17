# 14장 train.py: 13장의 한 파일을 패키지로 나눈 뒤 남은 것 = 흐름(main)만.
# 실행: python train.py --model mlp --dataset mnist --epochs 2 --run mlp-e2
import os
import time

import torch
import torch.nn as nn

from networks import build_model
from data import get_loaders
from utils import parse_train_args, evaluate, make_run_dir, save_json


def main():
    args = parse_train_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = make_run_dir(args.results, args.run)
    save_json(vars(args), os.path.join(out_dir, "config.json"))

    train_loader, test_loader = get_loaders(args.dataset, data_root=args.data_root, batch_size=args.batch_size)
    model = build_model(args.model, hidden=args.hidden).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    history, best_acc = [], 0.0
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
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
    torch.save(model.state_dict(), os.path.join(out_dir, "model_last.pt"))
    save_json({"best_test_acc": best_acc, "history": history}, os.path.join(out_dir, "metrics.json"))
    print(f"저장 완료: {out_dir}/")


if __name__ == "__main__":
    main()
