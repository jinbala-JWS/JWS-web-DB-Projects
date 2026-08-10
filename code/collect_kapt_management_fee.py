"""
K-apt(공동주택관리정보시스템) 공동주택관리비 수집기

아파트 유형만, 전국, 월별 "공용관리비"(원/㎡) 평균을 2012.01부터 수집한다.
CPI Tier A "공동주택관리비" 품목(가중치 21.8, 외부데이터 없음 상위 3위)의 외부지표 후보.

- 인증: 페이지의 <meta name="_csrf">를 읽어 폼 데이터 + X-CSRF-TOKEN 헤더 둘 다에 실어야 함
  (둘 중 하나만 보내면 세션에 따라 실패하는 경우가 있어 안전하게 둘 다 보냄).
- 공개 시차 확인됨: 발생월 데이터는 즉시 공개되지 않고 다음달부터 순차적으로 채워져서
  다다음달 말일에야 거의 완전해짐 (예: 2026-08-10 기준 2026-07 데이터는 321개 단지뿐인데
  2026-05는 19,295개 단지로 이미 안정화 -> 최근 2개월은 잠정치로 취급해야 함).
- 2011-12 이전은 데이터가 아예 없음(2012-01부터 시작).

사용법:
    python collect_kapt_management_fee.py                      # 2012-01 ~ 최근월-2개월까지
    python collect_kapt_management_fee.py --start 2012-01 --end 2026-05
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent / "processed"
OUT_CSV = OUT_DIR / "kapt_management_fee_monthly.csv"

BASE_URL = "https://www.k-apt.go.kr"
FORM_PAGE = f"{BASE_URL}/apiinfo/apiStatisticsSearch.do"
SEARCH_URL = f"{BASE_URL}/apiinfo/apiStatisticsSearchAvg.do"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 관심있는 상위 분류 항목 (columnNm1 기준)
FIELDS_OF_INTEREST = {
    "PUBLIC_TOT": "공용관리비",
    "PRIVATE_TOT": "개별사용료",
    "S_LEVY": "장기수선충당금_월부과액",
    "COST_TOT": "관리비총액",
    "OTHER_INCOME": "잡수입",
}


def month_range(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m = 1
            y += 1


def get_csrf(session: requests.Session) -> str:
    r = session.get(FORM_PAGE, headers=HEADERS, timeout=20)
    m = re.search(r'name="_csrf" content="([^"]+)"', r.text)
    if not m:
        raise RuntimeError("CSRF 토큰을 찾지 못함 - 페이지 구조가 바뀌었을 수 있음")
    return m.group(1)


def fetch_month(session: requests.Session, csrf: str, ym: str, kapt_type: str = "01") -> dict | None:
    sdate = ym.replace("-", "")
    data = {
        "gridheader_id": "", "kapt_type": kapt_type, "sale_type": "", "kapt_usedate": "",
        "kapt_usedate_range": "", "kapt_top_floor": "", "kapt_top_floor_range": "",
        "kaptda_cnt": "", "kaptda_cnt_range": "", "kapt_marea": "", "kapt_marea_range": "",
        "code_heat": "", "kapt_dong_cnt": "", "kapt_dong_cnt_range": "", "kaptd_pcnt": "",
        "kaptd_pcnt_range": "", "code_hall": "", "code_sec": "", "kaptd_scnt": "",
        "kaptd_scnt_range": "", "mgr_type": "", "kapt_mgr_cnt": "", "kapt_mgr_cnt_range": "",
        "kaptd_clcnt": "", "kaptd_clcnt_range": "", "code_clean": "", "bjd_code": "",
        "kaptd_search": "", "search_sdate": sdate, "search_edate": sdate, "_csrf": csrf,
    }
    headers = dict(HEADERS)
    headers["X-CSRF-TOKEN"] = csrf
    r = session.post(SEARCH_URL, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    try:
        j = r.json()
    except Exception:
        return None
    cnt = j.get("resultTotalCnt")
    if not cnt:
        return None
    row = {"date": ym, "n_complex": cnt}
    for item in j.get("resultList", []):
        col = item.get("columnNm1")
        if col in FIELDS_OF_INTEREST:
            row[FIELDS_OF_INTEREST[col]] = item.get("api")
    return row


def default_end_month() -> str:
    """가장 최근 2개월은 아직 공개가 덜 끝난 잠정치일 수 있어 기본값에서 제외."""
    today = date.today()
    y, m = today.year, today.month
    for _ in range(2):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return f"{y:04d}-{m:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2012-01")
    parser.add_argument("--end", default=default_end_month())
    parser.add_argument("--kapt-type", default="01", help="01=아파트(기본), 02=연립주택, 03=다세대 등")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    session = requests.Session()
    csrf = get_csrf(session)

    months = list(month_range(args.start, args.end))
    results = []
    for i, ym in enumerate(months, 1):
        try:
            row = fetch_month(session, csrf, ym, args.kapt_type)
            if row:
                results.append(row)
                fee = row.get("공용관리비")
                fee_str = f"{fee:.1f}원/㎡" if fee is not None else "N/A"
                print(f"[{i}/{len(months)}] {ym}: 단지수={row['n_complex']} 공용관리비={fee_str}")
            else:
                print(f"[{i}/{len(months)}] {ym}: 데이터 없음")
        except requests.exceptions.HTTPError:
            # CSRF 만료 시 재발급 후 1회 재시도
            csrf = get_csrf(session)
            row = fetch_month(session, csrf, ym, args.kapt_type)
            if row:
                results.append(row)
                print(f"[{i}/{len(months)}] {ym}: (재시도 성공) 단지수={row['n_complex']}")
        except Exception as e:
            print(f"[{i}/{len(months)}] {ym}: 오류 - {e}")
        time.sleep(0.2)

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n완료: {OUT_CSV} ({len(df)}행)")


if __name__ == "__main__":
    main()
