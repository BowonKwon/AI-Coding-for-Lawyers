# 2장 확인 스크립트: 지금 실행 중인 파이썬이 어느 환경의 것인지, PyTorch가 잘 설치됐는지 확인합니다.
# 실행: (jucos) 환경에서  python check_env.py

import sys          # 파이썬 실행 파일의 위치와 버전을 알려 주는 표준 모듈
import platform     # 운영체제 정보를 알려 주는 표준 모듈

print("파이썬 버전 :", sys.version.split()[0])
print("실행 파일   :", sys.executable)      # 경로에 'jucos'가 보여야 우리 환경입니다
print("운영체제    :", platform.system(), platform.release())

try:
    import torch                             # PyTorch 불러오기
    print("PyTorch 버전:", torch.__version__)
    print("GPU 사용 가능:", torch.cuda.is_available())  # 개인 PC는 대개 False. 정상입니다.
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print("텐서 연산 확인:", (x @ x).tolist())          # 행렬 곱 한 번
except ImportError:
    print("PyTorch가 아직 설치되지 않았습니다. 2.6절의 pip 명령을 먼저 실행하세요.")
