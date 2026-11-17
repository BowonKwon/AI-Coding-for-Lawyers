# data 패키지: 데이터셋과 로더를 만드는 함수를 모아 둔다.
from .mnist import get_mnist_loaders

LOADERS = {"mnist": get_mnist_loaders}     # 15장부터 cifar10, 내 이미지 폴더 등을 추가한다


def get_loaders(name, **kwargs):
    if name not in LOADERS:
        raise ValueError(f"모르는 데이터셋 {name!r}. 가능한 것: {list(LOADERS)}")
    return LOADERS[name](**kwargs)
