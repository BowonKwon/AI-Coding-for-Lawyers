# 4.6 종합 예제: CSV에서 사건을 읽어 객체로 만들고 보고서를 출력한다
# 3장의 case_records.py 를 함수·모듈·파일·예외·클래스로 다시 쓴 것.
# 실행: python case_report.py cases_all.csv   (파일이 없으면 만들어 줍니다)
import csv
import sys

from caselib import count_by                  # 4.2에서 만든 모듈 재사용


class Case:
    def __init__(self, case_no, case_type, claim, result, days):
        self.case_no = case_no
        self.type = case_type
        self.claim = claim
        self.result = result
        self.days = days

    @classmethod
    def from_row(cls, row):
        """CSV 한 행(딕셔너리, 값은 전부 글자)에서 Case를 만든다. 형식이 틀리면 ValueError."""
        return cls(
            row["case_no"],
            row["type"],
            int(row["claim"].replace(",", "")),
            row["result"],
            int(row["days"]),
        )

    def __repr__(self):
        return f"Case({self.case_no}, {self.type}, {self.claim:,}, {self.result}, {self.days}일)"


SAMPLE = [
    ["case_no", "type", "claim", "result", "days"],
    ["2099가합1001", "손해배상", "420000000", "인용", "310"],
    ["2099가합1002", "대여금", "150,000,000", "기각", "198"],      # 쉼표 있는 금액
    ["2099가합1003", "손해배상", "880000000", "일부인용", "455"],
    ["2099가합1004", "부당이득", "67000000", "인용", "120"],
    ["2099가합1005", "손해배상", "230000000", "기각", "260"],
    ["2099가합1006", "손해배상", "미정", "인용", "90"],            # 잘못된 행: 건너뛰어야 함
]


def ensure_sample(path):
    """예제 파일이 없으면 만든다."""
    try:
        open(path, encoding="utf-8-sig").close()
    except FileNotFoundError:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            csv.writer(f).writerows(SAMPLE)
        print(f"예제 파일을 만들었습니다: {path}")


def load_cases(path):
    cases, skipped = [], []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                cases.append(Case.from_row(row))
            except (ValueError, KeyError) as e:
                skipped.append((row.get("case_no", "?"), str(e)))
    return cases, skipped


def report(cases):
    records = [c.__dict__ for c in cases]     # 객체 → 딕셔너리 (caselib 함수에 넘기기 위해)
    print(f"사건 {len(cases)}건, 청구금액 합계 {sum(c.claim for c in cases):,}원")
    print("유형별 건수:", count_by(records, "type"))
    print("결과별 건수:", count_by(records, "result"))
    longest = max(cases, key=lambda c: c.days)
    print(f"최장 처리: {longest.case_no} ({longest.days}일)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "cases_all.csv"   # 명령줄 인자
    ensure_sample(path)
    cases, skipped = load_cases(path)
    report(cases)
    for case_no, why in skipped:
        print(f"건너뜀 {case_no}: {why}")
