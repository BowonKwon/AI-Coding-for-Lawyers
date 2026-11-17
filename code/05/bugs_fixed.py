# 5장 실습 정답: bugs.py 의 버그 셋을 고친 판.
# 버그 1: 네 번째 사건의 claim 이 글자 "67000000" (TypeError). → int() 로 변환하거나 데이터 수정.
# 버그 2: count_by_type 의 return 이 for 안에 들여 써져 첫 사건만 세고 끝남. → return 을 for 밖으로.
# 버그 3: mean_days 의 // 1 은 몫만 남겨 소수점을 버림. 반올림 의도였다면 round(). 여기서는 평균 그대로.

records = [
    {"case_no": "2099가합1001", "type": "손해배상", "claim": 420000000, "days": 310},
    {"case_no": "2099가합1002", "type": "대여금",   "claim": 150000000, "days": 198},
    {"case_no": "2099가합1003", "type": "손해배상", "claim": 880000000, "days": 455},
    {"case_no": "2099가합1004", "type": "부당이득", "claim": 67000000,  "days": 120},   # 버그 1 수정
    {"case_no": "2099가합1005", "type": "손해배상", "claim": 230000000, "days": 260},
]


def total_claim(records):
    total = 0
    for r in records:
        total += int(r["claim"])          # 글자가 섞여 들어와도 견디게
    return total


def count_by_type(records):
    counts = {}
    for r in records:
        t = r["type"]
        counts[t] = counts.get(t, 0) + 1
    return counts                          # 버그 2 수정: for 밖으로


def mean_days(records):
    total = 0
    for r in records:
        total += r["days"]
    return total / len(records)            # 버그 3 수정


print(f"사건 {len(records)}건, 청구금액 합계 {total_claim(records):,}원")
print("유형별 건수:", count_by_type(records))
print(f"평균 처리일수: {mean_days(records):.0f}일")
