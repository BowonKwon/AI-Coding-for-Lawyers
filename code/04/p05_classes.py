# 4.5 클래스와 객체: 데이터와 동작을 한 묶음으로

class Case:
    """사건 하나를 나타내는 틀(클래스)"""

    def __init__(self, case_no, claim, result=None):   # 객체를 만들 때 자동 실행
        self.case_no = case_no                          # self.이름 = 이 객체가 가진 값(속성)
        self.claim = claim
        self.result = result

    def year(self):                                     # 메서드: 이 객체가 할 수 있는 동작
        return int(self.case_no[:4])

    def is_decided(self):
        return self.result is not None

    def __repr__(self):                                 # print 할 때 보이는 모양
        return f"Case({self.case_no}, {self.claim:,}, {self.result})"


c1 = Case("2099가합1001", 420000000, "인용")   # 객체 만들기 = 클래스 이름을 함수처럼 호출
c2 = Case("2099가합1002", 150000000)          # result는 기본값 None
print(c1)
print(c1.case_no, c1.year(), c1.is_decided())
print(c2.is_decided())
c2.result = "기각"                            # 속성은 나중에 바꿀 수 있다
print(c2)


# ---- 상속: 기존 틀을 물려받아 조금만 바꾸기
class AppealCase(Case):                       # Case의 모든 것을 물려받는다
    def __init__(self, case_no, claim, result=None, lower_case_no=None):
        super().__init__(case_no, claim, result)       # 부모의 __init__ 먼저
        self.lower_case_no = lower_case_no

    def is_decided(self):                     # 부모 메서드를 덮어쓰기(override)
        return self.result in ("항소기각", "취소", "변경")


a1 = AppealCase("2099나5001", 420000000, "항소기각", lower_case_no="2099가합1001")
print(a1, a1.year(), a1.is_decided(), a1.lower_case_no)


# ---- 특수 메서드: 2부의 Dataset이 쓰는 두 가지
class CaseList:
    def __init__(self, cases):
        self.cases = cases

    def __len__(self):                        # len(obj) 가 호출한다
        return len(self.cases)

    def __getitem__(self, i):                 # obj[i] 가 호출한다
        return self.cases[i]


cl = CaseList([c1, c2, a1])
print(len(cl), cl[0], cl[-1])
for c in cl:                                  # __getitem__ 이 있으면 for 문도 된다
    print(" ", c.case_no)
