"""
KOSIS OpenAPI로 지출목적별 소비자물가지수(품목포함, 2020=100, DT_1J22001) 전 항목
(총지수+대분류12+소분류39+세분류71+세부품목458=581개)을 월 단위로 받아 저장하고,
학습 패널(cpi_official_monthly_wide.csv, all_tiers_monthly_panel.csv)에 해당 월 열을
반영한다.

사용: python fetch_kosis_cpi.py 202609 [202610 ...]
인증키는 저장소 밖 로컬 파일에서 읽는다(git에 올리지 않음).
"""
import re
import sys
from pathlib import Path

import pandas as pd
import requests

KEY_PATH = Path(r"C:\Users\infomax\Documents\DB for Claude\.credentials\kosis_api_key.txt")
SCRIPTS = Path(__file__).resolve().parent
URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


def fetch_month(yyyymm: str) -> pd.DataFrame:
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    params = dict(method="getList", apiKey=key, itmId="T+", objL1="T10", objL2="ALL",
                  format="json", jsonVD="Y", prdSe="M", startPrdDe=yyyymm, endPrdDe=yyyymm,
                  orgId="101", tblId="DT_1J22001")
    r = requests.get(URL, params=params, timeout=120)
    r.encoding = "utf-8"
    data = r.json()
    if isinstance(data, dict) and "err" in data:
        raise RuntimeError(f"KOSIS API 오류: {data}")
    df = pd.DataFrame(data)[["C2", "C2_NM", "PRD_DE", "DT", "LST_CHN_DE"]]
    df.columns = ["코드", "품목명", "기간", "지수", "KOSIS갱신일"]
    df["지수"] = pd.to_numeric(df["지수"], errors="coerce")
    df["레벨"] = df["코드"].map(lambda c: 0 if c == "0" else {1: 1, 3: 2, 4: 3}.get(len(c), 4))
    return df


def update_panels(months_df: pd.DataFrame, yyyymm: str):
    col = f"{yyyymm[:4]}-{yyyymm[4:]}"
    sub = months_df.sort_values("레벨")  # 세부품목(레벨4)이 같은 이름 충돌 시 마지막에 덮어씀
    norm = lambda x: re.sub(r"[\s,]+", "", str(x))  # API 응답은 쉼표가 빠져 있어 공백·쉼표 제거 후 매칭
    value_by_name = dict(zip(sub["품목명"].map(norm), sub["지수"]))

    for fname, name_idx in [("cpi_official_monthly_wide.csv", 0), ("all_tiers_monthly_panel.csv", 1)]:
        path = SCRIPTS / fname
        df = pd.read_csv(path, encoding="utf-8-sig")
        name_col = df.columns[name_idx]
        df[col] = df[name_col].map(norm).map(value_by_name)
        n_missing = int(df[col].isna().sum())
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[{fname}] {len(df)}행, {col} 미매칭 {n_missing}행")


def main():
    months = sys.argv[1:] or ["202609"]
    for m in months:
        df = fetch_month(m)
        out = SCRIPTS / f"kosis_cpi_{m}.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{m}: {len(df)}개 항목 저장 -> {out.name} (총지수 {df.loc[df['코드']=='0','지수'].iloc[0]})")
        update_panels(df, m)


if __name__ == "__main__":
    main()
