# networks 패키지: 모델 정의를 모아 둔다. 이 파일이 있어야 폴더가 패키지로 인식된다.
from .mlp import MLP

MODELS = {"mlp": MLP}          # 이름 → 클래스. 16장부터 여기에 lenet, vgg, ... 를 추가한다


def build_model(name, **kwargs):
    """이름으로 모델을 만든다. 예: build_model('mlp', hidden=128)"""
    if name not in MODELS:
        raise ValueError(f"모르는 모델 {name!r}. 가능한 것: {list(MODELS)}")
    return MODELS[name](**kwargs)
