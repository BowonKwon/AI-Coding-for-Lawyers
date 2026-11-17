import argparse
import os


def parse_train_args(argv=None):
    p = argparse.ArgumentParser(description="학습")
    p.add_argument("--model", default="mlp", help="networks.MODELS 의 이름")
    p.add_argument("--dataset", default="mnist", help="data.LOADERS 의 이름")
    p.add_argument("--data-root", default=os.environ.get("JUCOS_DATA", "./data"))
    p.add_argument("--results", default="results")
    p.add_argument("--run", default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args(argv)


def parse_infer_args(argv=None):
    p = argparse.ArgumentParser(description="추론")
    p.add_argument("--results", default="results")
    p.add_argument("--run", required=True)
    p.add_argument("--which", default="best", choices=["best", "last"])
    p.add_argument("--index", type=int, default=None)
    return p.parse_args(argv)
