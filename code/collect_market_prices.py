"""
Tier A 농축산물 외부가격 수집기

KAMIS(농산물유통정보)와 축산유통정보 다봄(축산물품질평가원)에서
매일 공시되는 "전국 평균 소매가격"을 가져와 code/processed/market_prices_daily.csv 에 누적 저장한다.

- 두 사이트 모두 회원가입/로그인 없이 조회 가능한 공공 통계이며, 할인행사가와는 별도로
  집계된 "평균 소매가"를 쓰므로 할인가 문제가 원천적으로 없다.
- 검증 결과(2026-08-10 기준): 쌀(KAMIS) vs CPI 쌀 지수 레벨상관 0.98, 전월비상관 0.86 /
  돼지고기(축산유통정보) vs CPI 돼지고기 지수 레벨상관 0.94, 전월비상관 0.89 — 둘 다 신뢰할 만함.

사용법:
    python collect_market_prices.py          # 오늘 날짜로 1행 수집/추가
    python collect_market_prices.py --date 2026-08-05   # 특정 날짜로 수집(과거 데이터 보강용)

매일 이 스크립트를 실행하면 CSV가 하루 1행씩 쌓여서, 나중에 전월비를 계산할 수 있다.
아직 자동 스케줄링은 설정하지 않았음 — 수동 실행 방식으로 몇 번 받아본 뒤 자동화 여부 결정.
"""

from __future__ import annotations

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

OUT_PATH = Path(__file__).resolve().parent / "processed" / "market_prices_daily.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ---------------------------------------------------------------------------
# 품목 설정 — CPI Tier A 품목명 -> 수집 소스/코드
# ---------------------------------------------------------------------------

# 축산유통정보(다봄): livestockType + spec(부위/규격) 조합. 단위는 원/100g (닭은 원/kg, 계란은 원/30개)
EKAPE_ITEMS = {
    "국산쇠고기": {"livestockType": "4301", "spec": "22", "unit": "원/100g(등심)"},
    "돼지고기":   {"livestockType": "4304", "spec": "27", "unit": "원/100g(삼겹살)"},
    "닭고기":     {"livestockType": "9901", "spec": "99", "unit": "원/kg(육계)"},
    "달걀":       {"livestockType": "9903", "spec": "23", "unit": "원/30구(특란)"},
    "우유":       {"livestockType": "9908", "spec": "01", "unit": "원(흰우유)"},
}

# KAMIS: itemcategorycode + itemcode. 단위는 등급/규격에 따라 다름(예: 쌀=20kg)
KAMIS_ITEMS = {
    "쌀":     {"itemcategorycode": "100", "itemcode": "111"},
    "찹쌀":   {"itemcategorycode": "100", "itemcode": "112"},
    "보리쌀": {"itemcategorycode": "100", "itemcode": "121"},
    "혼식곡": {"itemcategorycode": "100", "itemcode": "113"},
    "현미":   {"itemcategorycode": "100", "itemcode": "115"},
    "콩":     {"itemcategorycode": "100", "itemcode": "141"},
    "땅콩":   {"itemcategorycode": "300", "itemcode": "314"},
}


def fetch_ekape_price(livestock_type: str, spec: str, target_date: str) -> float | None:
    """축산유통정보(다봄)는 표가 서버 렌더링이 아니라 페이지에 박힌 JS 배열(chartData)로
    내려온다. 반드시 여러 날짜 범위로 조회해야(단일 날짜 범위는 빈 배열) 값이 채워진다."""
    end = target_date
    start = (date.fromisoformat(target_date) - timedelta(days=10)).isoformat()
    url = "https://www.ekapepia.com/v3/price/consumer/periodPrice.do"
    params = {
        "livestockType": livestock_type,
        "aggregationUnit": "DAY",
        "startDate": start,
        "endDate": end,
        "spec": spec,
        "grade": "",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    m = re.search(r"let chartData\s*=\s*\[(.*?)\]\s*\n\s*chartData = chartData\.reverse", r.text, re.S)
    if not m:
        return None
    entries = re.findall(r'\{"Month":"([\d/]+)",\s*"avg":"([\d.]*)"', m.group(1))
    if not entries:
        return None
    # 날짜(YY/MM/DD) 기준 가장 최근 항목
    entries_sorted = sorted(entries, key=lambda e: e[0], reverse=True)
    for _, avg in entries_sorted:
        if avg:
            return float(avg)
    return None


def fetch_kamis_price(itemcategorycode: str, itemcode: str, target_date: str) -> float | None:
    """KAMIS 소매가격(기간별/일간)은 requests로는 서버 500 오류가 나서 아직 자동화하지 못함
    (세션/파라미터 조합 문제로 추정, 미해결). 지금은 항상 None을 반환 -> 필요시 브라우저로
    수동 조회해서 processed/market_prices_daily.csv 에 직접 채워넣을 것."""
    return None


def collect(target_date: str) -> dict:
    result = {"date": target_date}
    print(f"[{target_date}] 축산유통정보 수집 중...")
    for name, cfg in EKAPE_ITEMS.items():
        try:
            val = fetch_ekape_price(cfg["livestockType"], cfg["spec"], target_date)
            result[name] = val
            print(f"   {name}: {val} ({cfg['unit']})")
        except Exception as e:
            print(f"   !! {name} 수집 실패: {e}")
            result[name] = None

    print(f"[{target_date}] KAMIS 수집 중...")
    for name, cfg in KAMIS_ITEMS.items():
        try:
            val = fetch_kamis_price(cfg["itemcategorycode"], cfg["itemcode"], target_date)
            result[name] = val
            print(f"   {name}: {val}")
        except Exception as e:
            print(f"   !! {name} 수집 실패: {e}")
            result[name] = None

    return result


def append_result(result: dict):
    OUT_PATH.parent.mkdir(exist_ok=True)
    row = pd.DataFrame([result])
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, encoding="utf-8-sig")
        existing = existing[existing["date"] != result["date"]]  # 같은 날짜 재실행시 갱신
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined = combined.sort_values("date")
    combined.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH} (총 {len(combined)}행)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat(), help="YYYY-MM-DD (기본: 오늘)")
    args = parser.parse_args()

    result = collect(args.date)
    append_result(result)
