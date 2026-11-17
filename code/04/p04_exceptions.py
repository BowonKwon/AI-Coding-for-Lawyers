# 4.4 예외 처리: 오류가 나도 프로그램이 죽지 않게

amount_texts = ["1200000", "350,000", "미정", "800000"]

# 처리하지 않으면 두 번째 값에서 프로그램이 멈춥니다
total = 0
for t in amount_texts:
    try:
        total += int(t)
    except ValueError:                        # 숫자로 못 바꾸는 경우만 잡는다
        print(f"건너뜀: {t!r}")               # !r: repr 형태로 출력
print("합계:", total)

# 쉼표 제거를 먼저 시도하고, 그래도 안 되면 건너뛰기
def to_int(text):
    try:
        return int(text.replace(",", ""))
    except ValueError:
        return None

print([to_int(t) for t in amount_texts])

# 여러 종류의 예외, else, finally
def read_first_line(path):
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        print("파일이 없습니다:", path)
        return None
    else:                                     # 예외가 없었을 때만
        line = f.readline()
        f.close()
        return line.strip()
    finally:                                  # 예외가 있든 없든 항상
        print("read_first_line 끝")

print(read_first_line("memo.txt"))
print(read_first_line("없는파일.txt"))

# 직접 예외 일으키기: 잘못된 입력을 초기에 걸러낸다
def parse_case_no(case_no):
    if len(case_no) < 7 or not case_no[:4].isdigit():
        raise ValueError(f"사건번호 형식이 아닙니다: {case_no!r}")
    return int(case_no[:4]), case_no[4:6], int(case_no[6:])

print(parse_case_no("2099가합1234"))
try:
    parse_case_no("가합1234")
except ValueError as e:
    print("오류:", e)
