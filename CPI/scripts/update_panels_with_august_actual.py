"""
9월 예측을 위한 선행작업 — 실제 8월 발표치(kosis_august_actual_full.json, 581개
전체 항목)를 두 핵심 패널에 반영한다. 이걸 안 하면 ETS가 7월까지만 보고 9월을
"2개월 앞" 예측하는 셈이 되어버린다.

- cpi_official_monthly_wide.csv: 대분류/소분류/세분류/세부품목 전체(공식 KOSIS 패널) '2026-08' 열 추가
- all_tiers_monthly_panel.csv: 458개 leaf 품목(바텀업 파이프라인 입력) '2026-08' 열 추가
"""
import json
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

with open(SCRIPTS / "kosis_august_actual_full.json", encoding="utf-8") as f:
    raw = json.load(f)
actual = {r["name"]: r["value"] for r in raw}  # 581개, 이미 들여쓰기 제거된 순수 품목명


def update_official():
    df = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = df.columns[0]
    df["2026-08"] = df[item_col].map(actual)
    n_missing = df["2026-08"].isna().sum()
    df.to_csv(SCRIPTS / "cpi_official_monthly_wide.csv", index=False, encoding="utf-8-sig")
    print(f"[cpi_official_monthly_wide.csv] {len(df)}행 중 2026-08 미매칭 {n_missing}행")
    return df


def update_all_tiers_panel():
    df = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv", encoding="utf-8-sig")
    name_col = df.columns[1]  # 품목명
    df["2026-08"] = df[name_col].map(actual)
    n_missing = df["2026-08"].isna().sum()
    df.to_csv(SCRIPTS / "all_tiers_monthly_panel.csv", index=False, encoding="utf-8-sig")
    print(f"[all_tiers_monthly_panel.csv] {len(df)}행 중 2026-08 미매칭 {n_missing}행")
    return df


if __name__ == "__main__":
    update_official()
    update_all_tiers_panel()
