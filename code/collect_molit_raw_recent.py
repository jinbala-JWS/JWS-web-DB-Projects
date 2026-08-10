"""
국토교통부 실거래가 - 최근 1년치 "원자료"(개별 거래건, 계약년월/계약일 포함) 수집기

collect_molit_rental.py는 월별 집계치만 저장하고 개별 거래일자를 버리기 때문에,
"전월 11일~당월 10일" 같은 달력월과 다른 기준으로 재집계하려면 원자료가 필요하다.
이 스크립트는 최근 ~14개월치(전월세 11~10일 윈도우 약 1년분을 만들기 위한 여유분)만
개별거래 원자료 그대로 processed/molit_raw/{유형코드}_{YYYY-MM}.csv 로 저장한다.

collect_molit_rental.py의 월별요약 진행상황(already_done)과는 무관하게 동작한다
(이미 요약해서 저장한 달이어도 원자료 파일이 없으면 다시 받는다).

일일 다운로드 한도(100건)를 공유하므로, run_molit_daily.ps1에서 이 스크립트를
먼저 실행해 원자료 확보를 우선순위로 두고, 남은 한도로 옛날 달 백필을 이어가도록 구성했다.

사용법:
    python collect_molit_raw_recent.py                       # 기본: 최근 14개월, A/B/C 전체
    python collect_molit_raw_recent.py --months 14 --max-requests 45
"""

from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path

import collect_molit_rental as base

OUT_DIR = base.OUT_DIR
RAW_DIR = OUT_DIR / "molit_raw"


def recent_months(n: int) -> list[str]:
    today = date.today()
    y, m = today.year, today.month
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    return list(reversed(out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=14,
                         help="최근 몇 개월치 원자료를 받을지 (11~10일 윈도우 1년분 만들려면 최소 13)")
    parser.add_argument("--types", default="A,B,C")
    parser.add_argument("--max-requests", type=int, default=45)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    months = recent_months(args.months)
    types = args.types.split(",")

    todo = []
    for t in types:
        for ym in months:
            path = RAW_DIR / f"{t}_{ym}.csv"
            if not path.exists():
                todo.append((t, ym))

    total = len(todo)
    print(f"원자료 대상 {len(types)*len(months)}건 중 이미 확보 {len(types)*len(months)-total}건, 남은 {total}건")
    if total == 0:
        print("모두 확보되어 있습니다.")
        return

    n_requests = 0
    for thing_no, ym in todo:
        if n_requests >= args.max_requests:
            print(f"\n이번 실행 한도({args.max_requests}건) 도달 - 다음에 이어서 실행하세요.")
            break
        n_requests += 1
        try:
            df = base.fetch_month(thing_no, ym)
            if df is not None and len(df):
                df.to_csv(RAW_DIR / f"{thing_no}_{ym}.csv", index=False, encoding="utf-8-sig")
                print(f"[{n_requests}/{min(args.max_requests,total)}] {base.THING_TYPES[thing_no]} {ym}: {len(df)}건 저장")
            else:
                # 빈 결과도 "확인 완료" 표시로 빈 파일을 남겨 재요청 낭비를 막는다
                import pandas as pd
                pd.DataFrame().to_csv(RAW_DIR / f"{thing_no}_{ym}.csv", index=False, encoding="utf-8-sig")
                print(f"[{n_requests}/{min(args.max_requests,total)}] {base.THING_TYPES[thing_no]} {ym}: 데이터 없음")
        except Exception as e:
            msg = str(e)
            print(f"[{n_requests}/{min(args.max_requests,total)}] {base.THING_TYPES[thing_no]} {ym}: 오류 - {msg}")
            if "다운로드 횟수" in msg or "100" in msg:
                print("일일 다운로드 한도에 걸린 것으로 보입니다. 여기서 중단합니다.")
                break
        time.sleep(0.4)

    done = len(list(RAW_DIR.glob("*.csv")))
    print(f"\n현재까지 원자료 파일 수: {done}개 ({RAW_DIR})")


if __name__ == "__main__":
    main()
