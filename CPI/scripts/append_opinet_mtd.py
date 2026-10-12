"""
일별 수집 로그(daily_price_log.csv, 오피넷_기준일 기준)에서 지정월의 MTD(월초~현재) 평균을
계산해 raw_opinet_gasoline_diesel_kerosene.tsv / raw_opinet_auto_lpg.tsv에 해당 월 행으로
넣는다(이미 그 월 행이 있으면 교체 - 재실행해도 안전). 월중 예측(상반월 등)에서 오피넷
회귀변수의 '당월 실측' 값으로 쓰인다.

사용: python append_opinet_mtd.py 2026-10
"""
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
LOG = SCRIPTS.parent / "일별가격추적" / "daily_price_log.csv"


def mtd(month: str):
    y, m = month.split("-")
    df = pd.read_csv(LOG, encoding="utf-8-sig")
    df.columns = ["수집일자", "오피넷기준일", "고급휘발유", "보통휘발유", "자동차용경유", "실내등유", "자동차용LPG", "원달러환율"]
    sel = df[df["오피넷기준일"].str.contains(f"{y}년{m}월", na=False)].copy()
    if sel.empty:
        raise SystemExit(f"{month} 일별 데이터 없음")
    days = pd.to_datetime(sel["오피넷기준일"].str.extract(r"(\d{4})년(\d{2})월(\d{2})일").agg("-".join, axis=1))
    return sel, days.min().strftime("%m/%d"), days.max().strftime("%m/%d")


def replace_row(path: Path, key_regex: str, new_line: str):
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if not re.match(key_regex, l)]
    lines.append(new_line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    month = sys.argv[1] if len(sys.argv) > 1 else None
    if not month:
        raise SystemExit("사용: python append_opinet_mtd.py YYYY-MM")
    y, m = month.split("-")
    sel, d0, d1 = mtd(month)
    n = len(sel)
    g, d, k, lpg = (sel[c].astype(float).mean() for c in ["보통휘발유", "자동차용경유", "실내등유", "자동차용LPG"])

    replace_row(SCRIPTS / "raw_opinet_gasoline_diesel_kerosene.tsv", rf"^{y}년{m}월",
                f"{y}년{m}월(부분월,{d0}~{d1} 일별평균 {n}일,daily_price_log.csv 기준)\t-\t{g:.2f}\t{d:.2f}\t{k:.2f}\t-")
    replace_row(SCRIPTS / "raw_opinet_auto_lpg.tsv", rf"^{y}-{m}",
                f"{y}-{m}(부분월,{d0}~{d1} {n}일평균,daily_price_log.csv 기준)\t{lpg:.2f}")
    print(f"{month} MTD({d0}~{d1}, {n}일): 휘발유 {g:.2f} 경유 {d:.2f} 등유 {k:.2f} 자동차용LPG {lpg:.2f}")


if __name__ == "__main__":
    main()
