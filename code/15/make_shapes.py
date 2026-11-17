# 15.3 실습용 이미지 폴더를 만든다: shapes/{circle,square,triangle}/*.png
# 실제 사진 대신 도형을 그려 쓰는 이유: 저작권·용량 문제가 없고, 무엇이 정답인지 분명하다.
import os
import random
from PIL import Image, ImageDraw

ROOT = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "shapes")
CLASSES = ["circle", "square", "triangle"]
N_PER_CLASS = {"train": 200, "test": 50}
SIZE = 48


def draw(kind, rng):
    img = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    d = ImageDraw.Draw(img)
    color = tuple(rng.randint(0, 180) for _ in range(3))
    r = rng.randint(10, 18)
    cx, cy = rng.randint(r + 2, SIZE - r - 2), rng.randint(r + 2, SIZE - r - 2)
    if kind == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif kind == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    return img


def main():
    rng = random.Random(0)
    total = 0
    for split, n in N_PER_CLASS.items():
        for kind in CLASSES:
            folder = os.path.join(ROOT, split, kind)
            os.makedirs(folder, exist_ok=True)
            for i in range(n):
                draw(kind, rng).save(os.path.join(folder, f"{kind}_{i:03d}.png"))
                total += 1
    print(f"{total}장 생성: {ROOT}/{{train,test}}/{{{','.join(CLASSES)}}}/")


if __name__ == "__main__":
    main()
