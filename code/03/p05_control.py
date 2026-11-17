# 3.9~3.10 조건문과 반복문
# 아래 기준 금액과 이율은 설명을 위한 가정값입니다. 실제 규정과 다릅니다.

# ---- 조건문
claim = 350000000
if claim > 500000000:
    print("A 유형")
elif claim > 100000000:
    print("B 유형")
else:
    print("C 유형")

is_final = False
has_appeal = True
if not is_final and has_appeal:       # and, or, not 으로 조건을 잇습니다
    print("상소심 계속 중")

# ---- for: 묶음을 하나씩 꺼내며 반복
amounts = [3000, 1200, 5400, 800]
total = 0
for a in amounts:
    total = total + a
print("합계:", total)

for i, a in enumerate(amounts):       # 번호와 값을 함께
    print(f"{i + 1}번째: {a}")

for n in range(3):                    # 0, 1, 2
    print("range:", n)

# ---- 조건과 반복을 함께
big = []
for a in amounts:
    if a >= 3000:
        big.append(a)
print("3000 이상:", big)

# 같은 일을 한 줄로 (리스트 컴프리헨션)
print("3000 이상:", [a for a in amounts if a >= 3000])

# ---- while: 조건이 참인 동안 반복
balance = 1000
years = 0
while balance < 2000:
    balance = balance * 1.05          # 가정 이율 5%
    years += 1                        # years = years + 1 과 같습니다
print(f"{years}년 뒤 {balance:.0f}")

# ---- break, continue
for a in amounts:
    if a < 1000:
        continue                      # 이번 차례는 건너뜀
    if a > 5000:
        break                         # 반복을 아예 끝냄
    print("처리:", a)
