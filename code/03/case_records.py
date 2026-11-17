# 3.11 종합 예제: 사건 목록 집계
# 아래 데이터는 설명을 위해 지어낸 가상의 사건입니다.

records = [
    {"case_no": "2099가합1001", "type": "손해배상", "claim": 420000000, "result": "인용",    "days": 310},
    {"case_no": "2099가합1002", "type": "대여금",   "claim": 150000000, "result": "기각",    "days": 198},
    {"case_no": "2099가합1003", "type": "손해배상", "claim": 880000000, "result": "일부인용", "days": 455},
    {"case_no": "2099가합1004", "type": "부당이득", "claim": 67000000,  "result": "인용",    "days": 120},
    {"case_no": "2099가합1005", "type": "손해배상", "claim": 230000000, "result": "기각",    "days": 260},
]

# 1) 전체 건수와 청구금액 합계
total_claim = 0
for r in records:
    total_claim += r["claim"]
print(f"사건 {len(records)}건, 청구금액 합계 {total_claim:,}원")

# 2) 사건 유형별 건수
count_by_type = {}
for r in records:
    t = r["type"]
    count_by_type[t] = count_by_type.get(t, 0) + 1
print("유형별 건수:", count_by_type)

# 3) 결과별 평균 처리일수
days_by_result = {}
for r in records:
    days_by_result.setdefault(r["result"], []).append(r["days"])
for result, days in days_by_result.items():
    print(f"{result}: 평균 {sum(days) / len(days):.0f}일 ({len(days)}건)")

# 4) 조건에 맞는 사건 고르기: 손해배상이면서 청구금액 3억 원 이상
selected = [r["case_no"] for r in records if r["type"] == "손해배상" and r["claim"] >= 300000000]
print("손해배상·3억 이상:", selected)

# 5) 청구금액 순 정렬 (큰 순)
for r in sorted(records, key=lambda r: r["claim"], reverse=True):
    print(f"  {r['case_no']}  {r['claim']:>13,}원  {r['result']}")
