# 16.3 forward 를 한 줄씩 지나며 shape 이 어떻게 바뀌는지 추적한다 (5장 디버거로 보던 것을 코드로)
import torch
import torch.nn.functional as F
from lenet import LeNet

m = LeNet()
x = torch.zeros(64, 1, 28, 28)
print("입력            ", tuple(x.shape))
x = m.conv1(x);               print("conv1 (5x5)     ", tuple(x.shape), "  28-5+1 = 24")
x = F.max_pool2d(F.relu(x), 2); print("relu+pool       ", tuple(x.shape), "  24/2 = 12")
x = m.conv2(x);               print("conv2 (5x5)     ", tuple(x.shape), "  12-5+1 = 8")
x = F.max_pool2d(F.relu(x), 2); print("relu+pool       ", tuple(x.shape), "  8/2 = 4")
x = x.flatten(1);             print("flatten(1)      ", tuple(x.shape), "  16*4*4 = 256")
x = F.relu(m.fc1(x));         print("fc1             ", tuple(x.shape))
x = F.relu(m.fc2(x));         print("fc2             ", tuple(x.shape))
x = m.fc3(x);                 print("fc3 (출력)      ", tuple(x.shape))

# 8장 mnist_main.py 의 Net 이 fc1 입력을 9216 으로 둔 이유도 같은 계산이다:
# conv1(3x3) 28→26, conv2(3x3) 26→24, pool 24→12, 64채널 → 64*12*12 = 9216
print("8장 Net 의 fc1 입력:", 64 * 12 * 12)
