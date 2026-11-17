# 4.1 함수: 이름 붙인 작업 묶음

def yearly_interest(principal, rate):        # 매개변수 두 개
    """원금과 연이율로 1년 이자를 계산한다."""   # 설명문(docstring). 없어도 되지만 붙이는 습관
    return principal * rate                   # 결과를 돌려준다

print(yearly_interest(100000000, 0.05))       # 인자 두 개를 넣어 호출
print(yearly_interest(rate=0.03, principal=5000000))   # 이름을 붙여 순서 무관

def interest(principal, rate=0.05, years=1):  # 기본값이 있는 매개변수
    return principal * rate * years

print(interest(100000000))                    # rate, years는 기본값
print(interest(100000000, years=3))           # 일부만 지정

def split_case_no(case_no):                   # 값을 여러 개 돌려주기 (튜플)
    return case_no[:4], case_no[4:6], case_no[6:]

year, code, number = split_case_no("2099가합1234")
print(year, code, number)

def describe(case_no, *tags):                 # 개수가 정해지지 않은 인자
    print(case_no, "태그:", tags)

describe("2099가합1234")
describe("2099가합1234", "손해배상", "항소")

# 돌려주는 값이 없는 함수는 None을 돌려준다
def greet(name):
    print(f"{name}님, 안녕하세요")

result = greet("홍길동")
print(result)

# 함수 안에서 만든 변수는 밖에서 보이지 않는다 (지역 변수)
def make_total(amounts):
    total = 0
    for a in amounts:
        total += a
    return total

print(make_total([1, 2, 3]))
try:
    print(total)
except NameError as e:
    print("밖에서는 못 씀:", e)
