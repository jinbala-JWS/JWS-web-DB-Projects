"""전체 458개 품목(Tier A+B+C+D) 2000-01~2026-07 월별 지수 패널 구성."""
import pandas as pd
from pathlib import Path

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"


def main():
    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv")
    official = official.drop_duplicates(subset="품목", keep="first")

    tiers = pd.read_csv(CPI_DIR / "CPI_Tier_분류.csv", encoding="utf-8-sig")
    tiers["가중치"] = pd.to_numeric(tiers["가중치"], errors="coerce")

    merged = tiers.merge(official, left_on="품목명", right_on="품목", how="left").drop(columns=["품목"])
    print("전체 병합 형태:", merged.shape)
    print("2020-01 결측:", merged["2020-01"].isna().sum())

    # 총지수(실제 공식 발표 총지수) 별도 저장 - 검증 기준값
    total = official[official["품목"] == "0 총지수"]
    total.to_csv(SCRIPTS / "official_total_index.csv", index=False, encoding="utf-8-sig")

    merged.to_csv(SCRIPTS / "all_tiers_monthly_panel.csv", index=False, encoding="utf-8-sig")
    print("저장:", SCRIPTS / "all_tiers_monthly_panel.csv")
    print(merged.groupby("Tier")["가중치"].agg(["count", "sum"]))


if __name__ == "__main__":
    main()
