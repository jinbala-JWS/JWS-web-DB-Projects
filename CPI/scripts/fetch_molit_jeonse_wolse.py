"""
국토교통부 실거래가공개시스템(rt.molit.go.kr) 전월세 실거래가 수집.
아파트/연립다세대/단독다가구 x 85㎡초과~102㎡이하 x 전국 x 월별(2026-02~현재)
브라우저 없이 순수 requests로 CSV 다운로드 폼 제출 재현.

주의: 전국(시도=전체) 조회는 계약일자 범위가 최대 1개월로 제한되어 있어 월별로 나눠서 수집.
"""
import calendar
import io
from datetime import date
from pathlib import Path

import pandas as pd
import requests

SCRIPTS = Path(__file__).resolve().parent
OUT_DIR = SCRIPTS / "molit_raw"
OUT_DIR.mkdir(exist_ok=True)

BASE_URL = "https://rt.molit.go.kr/pt/xls/xls.do?mobileAt="
DOWN_URL = "https://rt.molit.go.kr/pt/xls/ptXlsCSVDown.do"
HEADERS = {"User-Agent": "Mozilla/5.0"}

THING_TYPES = {"A": "아파트", "B": "연립다세대", "C": "단독다가구"}
MONTHS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]


def month_range(ym: str):
    y, m = int(ym[:4]), int(ym[5:7])
    last_day = calendar.monthrange(y, m)[1]
    today = date.today()
    end = date(y, m, last_day)
    if end > today:
        end = today  # 진행중인 달(8월)은 오늘까지만
    return f"{y}-{m:02d}-01", end.isoformat()


def fetch_one(session, thing_no, ym):
    frm, to = month_range(ym)
    data = {
        "srhThingNo": thing_no, "srhDelngSecd": "2", "srhAddrGbn": "1", "srhLfstsSecd": "1",
        "sidoNm": "전체", "sggNm": "전체", "emdNm": "전체", "loadNm": "전체",
        "areaNm": "85㎡초과~102㎡이하", "hsmpNm": "전체", "mobileAt": "",
        "srhFromDt": frm, "srhToDt": to,
        "srhNewRonSecd": "", "srhSidoCd": "", "srhSggCd": "", "srhEmdCd": "", "srhRoadNm": "",
        "srhLoadCd": "", "srhHsmpCd": "", "srhArea": "", "srhFromAmount": "", "srhToAmount": "",
        "srhLrArea": "3",
    }
    r = session.post(DOWN_URL, data=data, headers={**HEADERS, "Referer": BASE_URL}, timeout=60)
    r.raise_for_status()
    text = r.content.decode("cp949", errors="replace")
    lines = text.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith('"NO"'))
    csv_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_text))
    df["부동산유형"] = THING_TYPES[thing_no]
    df["수집년월"] = ym
    return df


def main():
    session = requests.Session()
    session.get(BASE_URL, headers=HEADERS, timeout=20)

    all_frames = []
    for thing_no, thing_name in THING_TYPES.items():
        for ym in MONTHS:
            try:
                df = fetch_one(session, thing_no, ym)
                print(f"{thing_name} {ym}: {len(df)}건")
                all_frames.append(df)
                out_path = OUT_DIR / f"{thing_no}_{ym}.csv"
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
            except Exception as e:
                print(f"{thing_name} {ym}: 실패 - {e}")

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(SCRIPTS / "molit_jeonse_wolse_raw_full.csv", index=False, encoding="utf-8-sig")
    print("\n전체 저장:", full.shape)


if __name__ == "__main__":
    main()
