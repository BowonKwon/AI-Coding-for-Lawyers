# 4.2 모듈: 이 파일 자체가 모듈입니다. 다른 파일에서 import caselib 로 불러 씁니다.

def parse_case_no(case_no):
    """'2099가합1234' -> (2099, '가합', 1234)"""
    year = int(case_no[:4])
    number = int(case_no[6:])
    code = case_no[4:6]
    return year, code, number


def total_claim(records):
    """사건 목록의 청구금액 합계"""
    return sum(r["claim"] for r in records)


def count_by(records, key):
    """records를 key 값별로 센다. 예: count_by(records, 'type')"""
    counts = {}
    for r in records:
        counts[r[key]] = counts.get(r[key], 0) + 1
    return counts


if __name__ == "__main__":            # 이 파일을 직접 실행했을 때만 아래가 돈다
    print("caselib 자체 실행:", parse_case_no("2099가합1234"))
