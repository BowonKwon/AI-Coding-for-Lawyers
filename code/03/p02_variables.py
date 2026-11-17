# 3.3 변수와 자료형

case_no = "2099가합1234"      # 문자열(str): 따옴표로 감싼 글자
claim = 350000000             # 정수(int): 청구금액 3억 5천만 원
rate = 0.05                   # 실수(float): 연 이율 5%
is_final = False              # 불리언(bool): 확정 여부. True 또는 False만 가능

print(case_no, claim, rate, is_final)
print(type(case_no), type(claim), type(rate), type(is_final))

# 계산과 갱신
interest = claim * rate       # 1년치 이자
print("1년 이자:", interest)
claim = claim + 10000000      # 변수는 다시 채울 수 있습니다 (청구취지 확장)
print("확장 후 청구금액:", claim)

# 형 변환
amount_text = "1200000"       # 서류에서 읽어 온 금액은 글자입니다
print(amount_text + "원")     # 글자끼리는 이어 붙습니다
print(int(amount_text) + 1)   # 숫자로 바꿔야 계산됩니다
print(str(claim) + "원")      # 숫자를 글자로

# f-string: 글자 안에 변수 값을 끼워 넣기
print(f"{case_no} 사건의 청구금액은 {claim:,}원, 이자는 {interest:,.0f}원입니다.")
