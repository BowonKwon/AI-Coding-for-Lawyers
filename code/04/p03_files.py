# 4.3 파일 읽고 쓰기 (텍스트, CSV, JSON)
import csv
import json

# ---- 텍스트 파일 쓰기와 읽기
with open("memo.txt", "w", encoding="utf-8") as f:     # w: 새로 쓰기 (있으면 덮어씀)
    f.write("첫 줄\n")
    f.write("둘째 줄\n")

with open("memo.txt", "r", encoding="utf-8") as f:     # r: 읽기
    text = f.read()
print(repr(text))                                       # repr: 줄바꿈 문자까지 보이게

with open("memo.txt", encoding="utf-8") as f:           # 한 줄씩
    for line in f:
        print("줄:", line.strip())

# ---- CSV: 표 형식. 엑셀에서 열리는 가장 단순한 형식
records = [
    {"case_no": "2099가합1001", "type": "손해배상", "claim": 420000000},
    {"case_no": "2099가합1002", "type": "대여금", "claim": 150000000},
]
with open("cases.csv", "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig: 엑셀 한글 깨짐 방지
    writer = csv.DictWriter(f, fieldnames=["case_no", "type", "claim"])
    writer.writeheader()
    writer.writerows(records)

with open("cases.csv", encoding="utf-8-sig", newline="") as f:
    loaded = list(csv.DictReader(f))
print(loaded)
print(type(loaded[0]["claim"]))                          # CSV에서 읽은 값은 전부 글자!
print(sum(int(r["claim"]) for r in loaded))              # 숫자로 바꿔 합계

# ---- JSON: 딕셔너리·리스트를 그대로 저장하는 형식 (설정 파일, 결과 저장에 씀)
with open("cases.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)  # ensure_ascii=False: 한글 그대로

with open("cases.json", encoding="utf-8") as f:
    loaded2 = json.load(f)
print(loaded2[0]["claim"], type(loaded2[0]["claim"]))    # JSON은 숫자를 숫자로 보존
