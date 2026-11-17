# 5장 실습: 이 파일에는 버그가 세 개 있습니다. 오류 메시지가 나는 것은 하나뿐이고,
# 나머지 둘은 조용히 틀린 값을 냅니다. 디버거로 찾으세요.
# 기대하는 출력:
#   사건 5건, 청구금액 합계 1,747,000,000원
#   유형별 건수: {'손해배상': 3, '대여금': 1, '부당이득': 1}
#   평균 처리일수: 269일

records = [
    {"case_no": "2099가합1001", "type": "손해배상", "claim": 420000000, "days": 310},
    {"case_no": "2099가합1002", "type": "대여금",   "claim": 150000000, "days": 198},
    {"case_no": "2099가합1003", "type": "손해배상", "claim": 880000000, "days": 455},
    {"case_no": "2099가합1004", "type": "부당이득", "claim": "67000000", "days": 120},
    {"case_no": "2099가합1005", "type": "손해배상", "claim": 230000000, "days": 260},
]


def total_claim(records):
    total = 0
    for r in records:
        total += r["claim"]
    return total


def count_by_type(records):
    counts = {}
    for r in records:
        t = r["type"]
        counts[t] = counts.get(t, 0) + 1
        return counts


def mean_days(records):
    total = 0
    for r in records:
        total += r["days"]
    return total / len(records) // 1


print(f"사건 {len(records)}건, 청구금액 합계 {total_claim(records):,}원")
print("유형별 건수:", count_by_type(records))
print(f"평균 처리일수: {mean_days(records):.0f}일")
