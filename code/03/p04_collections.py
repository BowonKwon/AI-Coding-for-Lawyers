# 3.5~3.8 묶음 자료형: 리스트, 튜플, 딕셔너리, 집합

# ---- 리스트: 순서가 있고 바꿀 수 있는 묶음
parties = ["원고", "피고"]
parties.append("보조참가인")          # 끝에 추가
print(parties, len(parties))
print(parties[0], parties[-1])        # 첫째, 마지막
print(parties[1:])                    # 둘째부터 끝까지
parties[1] = "피고 주식회사 가나"      # 바꾸기
print(parties)
print("원고" in parties)              # 포함 여부

amounts = [3000, 1200, 5400, 800]
print(sorted(amounts))                # 정렬한 새 리스트 (원본은 그대로)
print(sum(amounts), max(amounts), min(amounts))

# ---- 튜플: 순서가 있지만 바꿀 수 없는 묶음
judgment_date = (2026, 9, 6)
year, month, day = judgment_date      # 언패킹: 한 번에 꺼내기
print(f"{year}. {month}. {day}. 선고")

# ---- 딕셔너리: 이름표(키)로 값을 찾는 묶음
case = {
    "case_no": "2099가합1234",
    "court": "가상지방법원",
    "claim": 350000000,
    "parties": ["원고", "피고"],      # 값으로 리스트도 들어갑니다
}
print(case["court"])                  # 키로 값 찾기
case["result"] = "일부 인용"          # 새 키 추가
print(case.get("judge", "미입력"))    # 없는 키는 get으로 안전하게
print(list(case.keys()))              # 키 목록
for key, value in case.items():       # 키와 값을 함께 순회
    print(f"  {key}: {value}")

# ---- 집합: 중복이 없고 순서도 없는 묶음
cited_a = {"민법 제750조", "민법 제751조", "민사소송법 제202조"}
cited_b = {"민법 제750조", "상법 제401조"}
print(cited_a & cited_b)              # 교집합: 둘 다 인용한 조문
print(cited_a | cited_b)              # 합집합
print(cited_a - cited_b)              # 차집합: A에만 있는 조문
print(len(set(["민법 제750조", "민법 제750조", "민법 제751조"])))  # 중복 제거
