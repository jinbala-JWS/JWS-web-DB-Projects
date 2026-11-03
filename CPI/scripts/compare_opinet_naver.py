"""
오피넷 일별 수집로그(daily_price_log.csv) vs 네이버 증권 유가(출처=한국석유공사 오피넷) 일별 비교.
네이버는 공개 JSON API(api.stock.naver.com/marketindex/energy/<코드>/prices)로 받는다.
네이버가 제공하는 오피넷 계열은 휘발유(OIL_GSL)·고급휘발유(OIL_HGSL)·경유(OIL_LO) 3종뿐이며
자동차용LPG·취사용LPG·등유는 네이버에 없다(전체 목록 api.stock.naver.com/marketindex/energy 확인).
사용: python compare_opinet_naver.py 2026-10
결과: opinet_vs_naver_{월}_{유종}.csv (date, naver, opinet_log, diff) + 요약 opinet_vs_naver_{월}_summary.csv
"""
import sys
from pathlib import Path
import pandas as pd
import requests

SCRIPTS = Path(__file__).resolve().parent
month = sys.argv[1] if len(sys.argv) > 1 else "2026-10"
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://stock.naver.com/"}
# 네이버 코드 -> 로그 컬럼명
ITEMS = {"OIL_GSL": "보통휘발유", "OIL_HGSL": "고급휘발유", "OIL_LO": "자동차용경유"}


def fetch_naver(code, pages=5):
    rows = []
    for page in range(1, pages + 1):
        rows += requests.get(f"https://api.stock.naver.com/marketindex/energy/{code}/prices?page={page}&pageSize=40",
                             headers=H, timeout=20).json()
    nv = pd.DataFrame(rows)
    nv["date"] = pd.to_datetime(nv["localTradedAt"].str[:10])
    nv["naver"] = nv["closePrice"].str.replace(",", "").astype(float)
    return nv[["date", "naver"]].drop_duplicates("date")


lg = pd.read_csv(SCRIPTS.parent / "일별가격추적" / "daily_price_log.csv", encoding="utf-8-sig")
lg.columns = ["수집일자", "오피넷기준일", "고급휘발유", "보통휘발유", "자동차용경유", "실내등유", "자동차용LPG", "원달러환율"]
d = lg["오피넷기준일"].str.extract(r"(\d{4})년(\d{2})월(\d{2})일")
lg["date"] = pd.to_datetime(d[0] + "-" + d[1] + "-" + d[2])

summary = []
for code, col in ITEMS.items():
    nv = fetch_naver(code)
    op = lg[["date", col]].rename(columns={col: "opinet_log"}).drop_duplicates("date")
    m = nv.merge(op, on="date", how="outer").sort_values("date")
    m = m[m["date"].dt.strftime("%Y-%m") == month].copy()
    m["diff"] = m["opinet_log"] - m["naver"]
    m.to_csv(SCRIPTS / f"opinet_vs_naver_{month}_{col}.csv", index=False, encoding="utf-8-sig")
    both = m.dropna(subset=["naver", "opinet_log"])
    summary.append({
        "유종": col, "네이버코드": code, "비교일수": len(both), "네이버일수": int(m["naver"].notna().sum()),
        "로그일수": int(m["opinet_log"].notna().sum()),
        "네이버평균": round(both["naver"].mean(), 3), "로그평균": round(both["opinet_log"].mean(), 3),
        "평균차": round(both["diff"].mean(), 3), "최대절대차": round(both["diff"].abs().max(), 2),
        "최대차일": both.loc[both["diff"].abs().idxmax(), "date"].strftime("%Y-%m-%d"),
        "0.01원이내일수": int((both["diff"].abs() <= 0.01).sum()),
        "0.2원초과일수": int((both["diff"].abs() > 0.2).sum()),
        "상관": round(both["naver"].corr(both["opinet_log"]), 6),
    })
s = pd.DataFrame(summary)
s.to_csv(SCRIPTS / f"opinet_vs_naver_{month}_summary.csv", index=False, encoding="utf-8-sig")
print(s.to_string(index=False))
