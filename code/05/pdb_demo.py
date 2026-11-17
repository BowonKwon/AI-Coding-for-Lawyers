# breakpoint() 는 VS Code 없이 터미널에서 쓰는 디버거(pdb)를 켭니다.
records = [{"claim": 100}, {"claim": 250}, {"claim": 50}]
total = 0
for r in records:
    breakpoint()          # 여기서 멈춥니다
    total += r["claim"]
print("합계:", total)
