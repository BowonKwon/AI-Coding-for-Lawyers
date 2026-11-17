# 19.0 텍스트 실습 데이터 만들기: 국가법령정보 OPEN API 에서 법령 조문을 받아 텍스트 파일로 저장한다.
# 법령·판결문은 저작권법 제7조에 따라 저작권 보호를 받지 않으므로 실습 데이터로 자유롭게 쓸 수 있다.
# 준비: https://open.law.go.kr 에서 OPEN API 이용 신청 → 발급된 ID 를 환경변수 LAW_OC 에 넣는다.
# 실행: python fetch_statutes.py            → JUCOS_DATA/statutes/{법령명}.txt (한 줄 = 조문 하나)
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

OC = os.environ.get("LAW_OC")
OUT = os.path.join(os.environ.get("JUCOS_DATA", "./data"), "statutes")
LAWS = ["민법", "형법", "민사소송법", "형사소송법", "상법", "행정소송법"]
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.law.go.kr/"}   # Referer 가 없으면 거부된다


def get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def find_mst(name):
    """법령명으로 검색해 현행 법령의 일련번호(MST)를 얻는다."""
    q = urllib.parse.urlencode({"OC": OC, "target": "law", "type": "JSON", "query": name, "display": 100, "sort": "lasc"})
    data = get_json(f"https://www.law.go.kr/DRF/lawSearch.do?{q}")
    laws = data.get("LawSearch", {}).get("law", [])
    if isinstance(laws, dict):
        laws = [laws]
    exact = [l for l in laws if l.get("법령명한글") == name and l.get("현행연혁코드") == "현행"]
    if not exact:
        raise RuntimeError(f"{name}: 검색 결과 없음 ({[l.get('법령명한글') for l in laws][:5]})")
    return exact[0]["법령일련번호"], exact[0].get("시행일자")


def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s or "").replace("\r", "").strip()


def fetch_articles(mst):
    """조문 본문을 (조문번호, 제목, 내용) 목록으로."""
    q = urllib.parse.urlencode({"OC": OC, "target": "law", "type": "JSON", "MST": mst})
    data = get_json(f"https://www.law.go.kr/DRF/lawService.do?{q}")
    units = data["법령"]["조문"]["조문단위"]
    if isinstance(units, dict):
        units = [units]
    out = []
    for u in units:
        if u.get("조문여부") != "조문":            # 장·절 제목 등은 건너뛴다
            continue
        body = strip_tags(u.get("조문내용", ""))
        # 항이 따로 있으면 이어 붙인다
        hangs = u.get("항", [])
        if isinstance(hangs, dict):
            hangs = [hangs]
        for h in hangs:
            body += " " + strip_tags(h.get("항내용", ""))
            hos = h.get("호", [])
            if isinstance(hos, dict):
                hos = [hos]
            for ho in hos:
                body += " " + strip_tags(ho.get("호내용", ""))
        body = re.sub(r"\s+", " ", body).strip()
        if body:
            out.append((u.get("조문번호"), u.get("조문제목") or "", body))
    return out


def main():
    if not OC:
        sys.exit("환경변수 LAW_OC 가 없습니다. open.law.go.kr 에서 발급받은 ID 를 넣으세요.")
    os.makedirs(OUT, exist_ok=True)
    for name in LAWS:
        if os.path.exists(os.path.join(OUT, f"{name}.txt")):
            print(f"{name:<8} 이미 있음 — 건너뜀")
            continue
        mst, eff = find_mst(name)
        arts = fetch_articles(mst)
        path = os.path.join(OUT, f"{name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for no, title, body in arts:
                f.write(body + "\n")
        chars = sum(len(a[2]) for a in arts)
        print(f"{name:<8} MST {mst}  시행 {eff}  조문 {len(arts):5d}개  {chars:8,}자  → {path}")
        time.sleep(1)


if __name__ == "__main__":
    main()
