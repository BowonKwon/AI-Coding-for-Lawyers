# 4.2 모듈 불러오기

import caselib                                # 같은 폴더의 caselib.py
from caselib import count_by                  # 필요한 것만 이름으로

print(caselib.parse_case_no("2099가합1234"))
records = [
    {"case_no": "2099가합1001", "type": "손해배상", "claim": 420000000},
    {"case_no": "2099가합1002", "type": "대여금", "claim": 150000000},
]
print(caselib.total_claim(records))
print(count_by(records, "type"))

# 표준 라이브러리: 파이썬에 처음부터 들어 있는 모듈들
import math
import random
import os
import datetime as dt                         # 별명(alias)

print(math.sqrt(2), math.floor(3.7))
random.seed(0)                                # 같은 난수를 다시 얻으려면 seed를 고정
print(random.choice(["원고", "피고"]), random.randint(1, 6))
print(os.getcwd())                            # 지금 작업 폴더
print(os.path.join("data", "cases.csv"))      # 운영체제에 맞는 경로 구분자로 이어 붙임
today = dt.date(2026, 9, 6)
print(today, today + dt.timedelta(days=14))   # 14일 뒤
