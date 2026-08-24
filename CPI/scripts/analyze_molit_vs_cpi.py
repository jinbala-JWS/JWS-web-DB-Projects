"""MOLIT 실거래가 월별 평균가 계산 + CPI 전세/월세 지수와 비교."""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(SCRIPTS / "molit_jeonse_wolse_raw_full.csv")
    df["보증금(만원)"] = pd.to_numeric(df["보증금(만원)"].astype(str).str.replace(",", ""), errors="coerce")
    df["월세금(만원)"] = pd.to_numeric(df["월세금(만원)"].astype(str).str.replace(",", ""), errors="coerce")

    rows = []
    for (thing, gubun), g in df.groupby(["부동산유형", "전월세구분"]):
        for ym, gm in g.groupby("수집년월"):
            n = len(gm)
            if gubun == "전세":
                avg = gm["보증금(만원)"].mean()
            else:
                avg = gm["월세금(만원)"].mean()
            rows.append({"부동산유형": thing, "구분": gubun, "년월": ym, "건수": n, "평균가(만원)": avg})

    summary = pd.DataFrame(rows).sort_values(["부동산유형", "구분", "년월"])
    summary.to_csv(SCRIPTS / "molit_monthly_avg.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("부동산유형 x 전세/월세 월별 평균가(만원) 및 2월 대비 누적상승률")
    print("=" * 80)
    growth_rows = []
    for (thing, gubun), g in summary.groupby(["부동산유형", "구분"]):
        g = g.set_index("년월")
        base = g["평균가(만원)"].iloc[0]
        base_month = g.index[0]
        print(f"\n[{thing} - {gubun}] (기준월 {base_month} = {base:,.0f}만원)")
        for ym, row in g.iterrows():
            pct = (row["평균가(만원)"] - base) / base * 100
            flag = " (진행중, 참고용)" if ym == "2026-08" else ""
            print(f"  {ym}: {row['평균가(만원)']:>10,.0f}만원 (건수{row['건수']:>5}) | {base_month}대비 {pct:+.2f}%{flag}")

        # 완결월 기준(2월~7월)이 주 지표, 8월은 진행중이라 참고치로 별도 표시
        full = g[g.index <= "2026-07"]
        last_full = full.iloc[-1]
        last_full_month = full.index[-1]
        total_pct = (last_full["평균가(만원)"] - base) / base * 100
        n_months = len(full) - 1
        mom_avg = total_pct / n_months if n_months else 0

        aug_pct = None
        if "2026-08" in g.index:
            aug_row = g.loc["2026-08"]
            aug_pct = (aug_row["평균가(만원)"] - base) / base * 100

        growth_rows.append({
            "부동산유형": thing, "구분": gubun,
            f"{base_month}_평균": base, f"{last_full_month}_평균(완결월기준)": last_full["평균가(만원)"],
            "누적상승률%(2~7월)": total_pct, "월평균상승률%": mom_avg,
            "8월(진행중)_참고상승률%": aug_pct,
        })

    growth_df = pd.DataFrame(growth_rows)
    growth_df.to_csv(SCRIPTS / "molit_growth_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("요약: 유형별 2월->7월(8월은 집계중이라 제외) 누적/월평균 상승률")
    print("=" * 80)
    print(growth_df.to_string(index=False))

    return summary, growth_df


if __name__ == "__main__":
    main()
