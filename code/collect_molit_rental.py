"""
국토교통부 실거래가 공개시스템 - 전월세 실거래가 수집기

아파트/연립·다세대/단독·다가구 3개 유형, 전용면적 85㎡초과~102㎡이하만,
2011.01(자료 공개 시작월) ~ 최신월까지 전국 데이터를 월별로 내려받아
전세(보증금)/월세(월세금) 평균·중앙값을 월별로 집계한다.

- 국토부 자료제공 페이지(rt.molit.go.kr/pt/xls/xls.do)가 "전국 자료제공 계약일자
  범위는 최대 1개월"이라 월 단위로 나눠서 순차 요청한다.
- 공공 서비스에 부담을 주지 않도록 요청 사이에 0.4초 대기를 둔다.
- 원자료(개별 거래건)는 용량이 커서 저장하지 않고, 월별 집계치만 저장한다.
  필요하면 --keep-raw 로 원자료도 함께 저장 가능.

사용법:
    python collect_molit_rental.py                    # 2011-01 ~ 최신월 전체 수집
    python collect_molit_rental.py --start 2020-01 --end 2024-12
"""

from __future__ import annotations

import argparse
import time
from calendar import monthrange
from pathlib import Path

import pandas as pd
import requests

OUT_DIR = Path(__file__).resolve().parent / "processed"
RAW_DIR = OUT_DIR / "molit_raw"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
URL = "https://rt.molit.go.kr/pt/xls/ptXlsCSVDown.do"

THING_TYPES = {"A": "아파트", "B": "연립다세대", "C": "단독다가구"}

BASE_PARAMS = {
    "srhDelngSecd": "2",  # 전월세
    "srhLrArea": "3",     # 85㎡초과~102㎡이하
    "srhArea": "",
    "srhSidoCd": "", "srhSggCd": "", "srhEmdCd": "", "srhHsmpCd": "", "srhLoadCd": "",
    "srhRoadNm": "", "srhAddrGbn": "1", "srhFromAmount": "", "srhToAmount": "",
    "srhNewRonSecd": "", "srhLfstsSecd": "1", "sidoNm": "", "sggNm": "", "emdNm": "",
    "hsmpNm": "", "areaNm": "85㎡초과~102㎡이하", "loadNm": "", "mobileAt": "",
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


def fetch_month(thing_no: str, ym: str) -> pd.DataFrame | None:
    y, m = map(int, ym.split("-"))
    last_day = monthrange(y, m)[1]
    data = dict(BASE_PARAMS, srhThingNo=thing_no,
                srhFromDt=f"{ym}-01", srhToDt=f"{ym}-{last_day:02d}")
    r = requests.post(URL, data=data, headers=HEADERS, timeout=60)
    r.raise_for_status()
    if r.content[:2] == b'{"':
        # {"error":"..."} 형태의 실패 응답 (일일 다운로드 한도 초과 등)
        raise RuntimeError(r.content.decode("utf-8", errors="replace"))
    text = r.content.decode("cp949", errors="replace")
    lines = text.splitlines()
    # 헤더(검색조건 설명) 다음, "NO","시군구",... 로 시작하는 실제 컬럼 헤더 행을 찾는다
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"NO","시군구"'):
            header_idx = i
            break
    if header_idx is None:
        return None  # 해당 월 데이터 없음
    import io
    csv_text = "\n".join(lines[header_idx:])
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None
    if df.empty:
        return None
    return df


def summarize(df: pd.DataFrame, ym: str, thing_no: str) -> dict:
    df = df.copy()
    for col in ["보증금(만원)", "월세금(만원)"]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)
    jeonse = df[df["전월세구분"] == "전세"]
    wolse = df[df["전월세구분"] == "월세"]
    return {
        "date": ym,
        "housing_type": THING_TYPES[thing_no],
        "jeonse_n": len(jeonse),
        "jeonse_deposit_mean": jeonse["보증금(만원)"].mean() if len(jeonse) else None,
        "jeonse_deposit_median": jeonse["보증금(만원)"].median() if len(jeonse) else None,
        "wolse_n": len(wolse),
        "wolse_deposit_mean": wolse["보증금(만원)"].mean() if len(wolse) else None,
        "wolse_rent_mean": wolse["월세금(만원)"].mean() if len(wolse) else None,
        "wolse_rent_median": wolse["월세금(만원)"].median() if len(wolse) else None,
    }


OUT_CSV = OUT_DIR / "molit_rental_monthly.csv"


def load_existing() -> pd.DataFrame:
    if OUT_CSV.exists():
        return pd.read_csv(OUT_CSV, encoding="utf-8-sig")
    return pd.DataFrame(columns=["date", "housing_type"])


def append_row(row: dict):
    """한 건씩 즉시 저장 -> 일일 다운로드 한도(100건)에 걸려 중단되어도 진행분은 안전하게 남는다."""
    existing = load_existing()
    existing = existing[~((existing["date"] == row["date"]) & (existing["housing_type"] == row["housing_type"]))]
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    combined = combined.sort_values(["housing_type", "date"])
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2011-01")
    parser.add_argument("--end", default="2026-07")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--types", default="A,B,C")
    parser.add_argument("--max-requests", type=int, default=95,
                         help="국토부 일일 다운로드 한도(100건)를 넘기지 않기 위한 이번 실행 최대 요청 수")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    if args.keep_raw:
        RAW_DIR.mkdir(exist_ok=True)

    months = list(month_range(args.start, args.end))
    types = args.types.split(",")
    existing = load_existing()
    already_done = set(zip(existing["date"], existing["housing_type"])) if len(existing) else set()

    todo = [(t, ym) for t in types for ym in months if (ym, THING_TYPES[t]) not in already_done]
    total = len(todo)
    print(f"전체 대상 {len(types)*len(months)}건 중 이미 완료 {len(already_done)}건, 남은 {total}건")
    if total == 0:
        print("모두 완료되어 있습니다.")
        return

    n_requests = 0
    for thing_no, ym in todo:
        if n_requests >= args.max_requests:
            print(f"\n이번 실행 한도({args.max_requests}건) 도달 - 다음에 이어서 실행하세요 "
                  f"(같은 명령 다시 실행하면 이어서 받아옴).")
            break
        n_requests += 1
        try:
            df = fetch_month(thing_no, ym)
            if df is not None and len(df):
                row = summarize(df, ym, thing_no)
                append_row(row)
                if args.keep_raw:
                    df.to_csv(RAW_DIR / f"{thing_no}_{ym}.csv", index=False, encoding="utf-8-sig")
                print(f"[{n_requests}/{min(args.max_requests,total)}] {THING_TYPES[thing_no]} {ym}: "
                      f"전세{len(df[df['전월세구분']=='전세'])}건 월세{len(df[df['전월세구분']=='월세'])}건")
            else:
                append_row({"date": ym, "housing_type": THING_TYPES[thing_no],
                            "jeonse_n": 0, "jeonse_deposit_mean": None, "jeonse_deposit_median": None,
                            "wolse_n": 0, "wolse_deposit_mean": None, "wolse_rent_mean": None, "wolse_rent_median": None})
                print(f"[{n_requests}/{min(args.max_requests,total)}] {THING_TYPES[thing_no]} {ym}: 데이터 없음(0건으로 기록)")
        except Exception as e:
            msg = str(e)
            print(f"[{n_requests}/{min(args.max_requests,total)}] {THING_TYPES[thing_no]} {ym}: 오류 - {msg}")
            if "다운로드 횟수" in msg or "100" in msg:
                print("일일 다운로드 한도에 걸린 것으로 보입니다. 여기서 중단합니다.")
                break
        time.sleep(0.4)  # 서버 부담 완화

    final = load_existing()
    print(f"\n현재까지 누적: {OUT_CSV} ({len(final)}행)")


if __name__ == "__main__":
    main()
