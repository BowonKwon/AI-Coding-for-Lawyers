# 20장 순환 신경망: 직접 만든 RNN 셀·LSTM 셀과 PyTorch 내장 nn.LSTM / nn.GRU
import torch
import torch.nn as nn


class MyRNNCell(nn.Module):
    """가장 단순한 순환 셀: h_t = tanh(W_x x_t + W_h h_{t-1} + b)"""

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.wx = nn.Linear(in_dim, hid_dim)
        self.wh = nn.Linear(hid_dim, hid_dim, bias=False)

    def forward(self, x_t, h_prev):
        return torch.tanh(self.wx(x_t) + self.wh(h_prev))


class MyLSTMCell(nn.Module):
    """LSTM 셀: 게이트 셋(잊기 f, 입력 i, 출력 o)과 후보 g. 상태는 (h, c) 둘."""

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.gates = nn.Linear(in_dim + hid_dim, 4 * hid_dim)   # 네 가지를 한 번에 계산
        self.hid_dim = hid_dim

    def forward(self, x_t, state):
        h_prev, c_prev = state
        z = self.gates(torch.cat([x_t, h_prev], dim=1))          # [B, 4H]
        i, f, g, o = z.chunk(4, dim=1)                            # 넷으로 나눔
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)   # 게이트는 0~1
        g = torch.tanh(g)                                          # 후보 값
        c = f * c_prev + i * g                                    # 기억: 잊을 만큼 잊고, 새것을 더한다
        h = o * torch.tanh(c)                                     # 출력: 기억 중 내보낼 만큼
        return h, c


class MyRNNClassifier(nn.Module):
    """임베딩 → 직접 만든 셀을 시간축으로 반복 → 마지막 h 로 분류"""

    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, cell="rnn"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.cell = MyRNNCell(emb_dim, hid_dim) if cell == "rnn" else MyLSTMCell(emb_dim, hid_dim)
        self.kind = cell
        self.fc = nn.Linear(hid_dim, num_classes)
        self.hid_dim = hid_dim

    def forward(self, ids):                                       # ids: [B, L]
        x = self.emb(ids)                                         # [B, L, E]
        B, L, _ = x.shape
        h = x.new_zeros(B, self.hid_dim)
        c = x.new_zeros(B, self.hid_dim)
        lengths = (ids != 0).sum(1)                               # 패딩 제외 실제 길이
        last_h = x.new_zeros(B, self.hid_dim)
        for t in range(L):                                        # 시간축으로 한 토큰씩
            if self.kind == "rnn":
                h = self.cell(x[:, t], h)
            else:
                h, c = self.cell(x[:, t], (h, c))
            is_last = (lengths == t + 1).unsqueeze(1)             # 이 토큰이 문장의 마지막이면
            last_h = torch.where(is_last, h, last_h)              # 그때의 h 를 저장
        return self.fc(last_h)


class TorchRNNClassifier(nn.Module):
    """같은 일을 PyTorch 내장 nn.LSTM / nn.GRU 로. 시간축 반복이 C++ 로 되어 훨씬 빠르다."""

    def __init__(self, vocab_size, emb_dim, hid_dim, num_classes, cell="lstm", bidirectional=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[cell]
        self.rnn = rnn_cls(emb_dim, hid_dim, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hid_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, ids):
        x = self.emb(ids)                                         # [B, L, E]
        lengths = (ids != 0).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)                                   # h: 마지막 은닉 상태 [layers*dirs, B, H]
        if isinstance(h, tuple):                                  # LSTM 은 (h, c)
            h = h[0]
        h = torch.cat([h[-2], h[-1]], dim=1) if self.rnn.bidirectional else h[-1]
        return self.fc(h)


if __name__ == "__main__":
    torch.manual_seed(0)
    ids = torch.tensor([[5, 3, 9, 0, 0], [7, 2, 4, 8, 1]])       # 배치 2, 길이 5 (첫 문장은 3토큰 + 패딩)
    for name, m in [("MyRNN", MyRNNClassifier(20, 8, 16, 3, "rnn")),
                    ("MyLSTM", MyRNNClassifier(20, 8, 16, 3, "lstm")),
                    ("nn.LSTM", TorchRNNClassifier(20, 8, 16, 3, "lstm")),
                    ("nn.GRU", TorchRNNClassifier(20, 8, 16, 3, "gru")),
                    ("BiLSTM", TorchRNNClassifier(20, 8, 16, 3, "lstm", bidirectional=True))]:
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:<8} 파라미터 {n:>6,}  [2,5] → {tuple(m(ids).shape)}")
    # 셀 하나의 계산: 입력 8, 은닉 16 일 때 LSTM 파라미터 = (8+16)*64 + 64
    print("MyLSTMCell 파라미터:", sum(p.numel() for p in MyLSTMCell(8, 16).parameters()), "= (8+16)*64+64")
