"""Tier A/B 418개 품목의 2000-01~2026-07 월별 지수 패널을 구성한다."""
import pandas as pd
from pathlib import Path

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"


def main():
    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv")
    official = official.drop_duplicates(subset="품목", keep="first")

    tiers = pd.read_csv(CPI_DIR / "CPI_Tier_분류.csv", encoding="utf-8-sig")
    tiers["가중치"] = pd.to_numeric(tiers["가중치"], errors="coerce")

    ab = tiers[tiers["Tier"].isin(["A", "B"])][["품목코드", "품목명", "가중치", "Tier", "세부"]]
    merged = ab.merge(official, left_on="품목명", right_on="품목", how="left").drop(columns=["품목"])

    missing = merged[merged.isna().any(axis=1) & merged["2020-01"].isna()]
    print("병합 후 형태:", merged.shape)
    print("2020-01 값이 없는 품목 수:", merged["2020-01"].isna().sum())

    merged.to_csv(SCRIPTS / "tierAB_monthly_panel.csv", index=False, encoding="utf-8-sig")
    print("저장:", SCRIPTS / "tierAB_monthly_panel.csv")


if __name__ == "__main__":
    main()
