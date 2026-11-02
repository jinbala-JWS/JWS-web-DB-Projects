"""
오피넷 일별 수집로그(daily_price_log.csv, 보통휘발유) vs 네이버 증권 휘발유(OIL_GSL, 출처=한국석유공사 오피넷)
일별 비교. 네이버는 공개 JSON API(api.stock.naver.com/marketindex/energy/OIL_GSL/prices)로 받는다.
사용: python compare_gasoline_naver.py 2026-10
"""
import sys
from pathlib import Path
import pandas as pd
import requests

SCRIPTS = Path(__file__).resolve().parent
month = sys.argv[1] if len(sys.argv) > 1 else "2026-10"
h = {"User-Agent": "Mozilla/5.0", "Referer": "https://stock.naver.com/"}
rows = []
for page in range(1, 6):
    rows += requests.get(f"https://api.stock.naver.com/marketindex/energy/OIL_GSL/prices?page={page}&pageSize=40",
                         headers=h, timeout=20).json()
nv = pd.DataFrame(rows)
nv["date"] = pd.to_datetime(nv["localTradedAt"].str[:10])
nv["naver"] = nv["closePrice"].str.replace(",", "").astype(float)
nv = nv[["date", "naver"]].drop_duplicates("date")

lg = pd.read_csv(SCRIPTS.parent / "일별가격추적" / "daily_price_log.csv", encoding="utf-8-sig")
lg.columns = ["수집일자", "오피넷기준일", "고급휘발유", "보통휘발유", "자동차용경유", "실내등유", "자동차용LPG", "원달러환율"]
d = lg["오피넷기준일"].str.extract(r"(\d{4})년(\d{2})월(\d{2})일")
lg["date"] = pd.to_datetime(d[0] + "-" + d[1] + "-" + d[2])
op = lg[["date", "수집일자", "보통휘발유"]].rename(columns={"보통휘발유": "opinet_log"}).drop_duplicates("date")

m = nv.merge(op, on="date", how="outer").sort_values("date")
m = m[m["date"].dt.strftime("%Y-%m") == month].copy()
m["diff"] = m["opinet_log"] - m["naver"]
m.to_csv(SCRIPTS / f"gasoline_opinet_vs_naver_{month}.csv", index=False, encoding="utf-8-sig")
print(f"{month}: naver {m['naver'].notna().sum()}일, log {m['opinet_log'].notna().sum()}일, "
      f"평균 naver {m['naver'].mean():.3f} / log {m['opinet_log'].mean():.3f}, 최대|차| {m['diff'].abs().max():.2f}원")
