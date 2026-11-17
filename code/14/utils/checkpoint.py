import json
import os
import time


def make_run_dir(results, run=None):
    """results/{run}/ 을 만들고 경로를 돌려준다. run 이 없으면 시각으로 이름 짓는다."""
    out_dir = os.path.join(results, run or time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
