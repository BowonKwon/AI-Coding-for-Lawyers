# 3.4 문자열 다루기

case_no = "2099가합1234"

# 인덱싱: 0부터 셉니다
print(case_no[0])        # 첫 글자
print(case_no[-1])       # 마지막 글자

# 슬라이싱: [시작:끝]  끝은 포함하지 않습니다
year = case_no[:4]       # 0,1,2,3번 글자
code = case_no[4:6]      # 4,5번 글자
number = case_no[6:]     # 6번부터 끝까지
print(year, code, number)

# 자주 쓰는 메서드
title = "  손해배상(기)  "
print(title.strip())                  # 양쪽 공백 제거
print("2099가합1234".startswith("2099"))  # 앞부분 확인
print("원고, 피고, 참가인".split(", "))  # 쉼표로 나누어 리스트로
print("가합".replace("합", "단"))       # 바꾸기
print(len(case_no))                   # 글자 수

# 검색과 포함
print("합" in case_no)   # 포함 여부는 True/False
print(case_no.find("합"))  # 위치(없으면 -1)
