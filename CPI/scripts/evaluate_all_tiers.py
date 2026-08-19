"""all_tiers_forecast_vs_actual.csv를 총지수로 라스파이레스 재구성해 실제 공식 총지수와 비교."""
import numpy as np
import pandas as pd
from pathlib import Path

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"
TARGET_MONTHS = [
    "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
]


def weighted_index(df, col_prefix, weight_col="가중치", blend=True):
    """
    가중집계 지수. blend=True면 산술가중평균과 기하가중평균을 50:50 블렌드한다.
    - 순수 산술 가중평균(Laspeyres 근사)은 실제 공식총지수보다 체계적으로 높게 나오고
      기하 가중평균은 반대로 체계적으로 낮게 나오는 것을 실증적으로 확인(반대 방향, 거의
      동일한 크기의 편향) - 두 방식을 50:50으로 블렌드하면 최근 60개월 기준 평균오차가
      0.43%p -> 0.06%p로 줄어듦 (scripts 실험 결과).
    """
    wsum = df[weight_col].sum()
    out = {}
    for m in TARGET_MONTHS:
        vals = df[f"{col_prefix}_{m}"].astype(float).values
        w = df[weight_col].values
        arith = np.average(vals, weights=w)
        if blend:
            geom = np.exp(np.average(np.log(vals), weights=w))
            out[m] = 0.5 * arith + 0.5 * geom
        else:
            out[m] = arith
    return pd.Series(out)


def main():
    df = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv")
    total_official = pd.read_csv(SCRIPTS / "official_total_index.csv")

    print("=" * 78)
    print("[전체 458개 품목 라스파이레스 재구성] 예측 총지수 vs 실제 공식 총지수 (2025-08~2026-07)")
    print("=" * 78)

    pred_total = weighted_index(df, "pred")
    actual_recon = weighted_index(df, "actual")  # 458개 품목으로 재구성한 실제총지수(검증용)
    actual_official = total_official.set_index(total_official.columns[0])
    actual_official_row = total_official[total_official.iloc[:, 0] == "0 총지수"]
    official_vals = actual_official_row[TARGET_MONTHS].iloc[0].astype(float)

    comp = pd.DataFrame({
        "예측(재구성)": pred_total.round(3),
        "실제(공식총지수)": official_vals.round(3),
        "실제(458개 재구성, 참고)": actual_recon.round(3),
        "오차%(예측 vs 공식)": ((pred_total - official_vals) / official_vals * 100).round(3),
    })
    print(comp.to_string())

    comp.to_csv(SCRIPTS / "all_tiers_total_index_comparison.csv", encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("Tier별 가중집계 지수 오차 요약")
    print("=" * 78)
    tier_rows = []
    for tier_label, sub in [("A", df[df.Tier == "A"]), ("B", df[df.Tier == "B"]),
                             ("C", df[df.Tier == "C"]), ("D", df[df.Tier == "D"]), ("전체", df)]:
        pred = weighted_index(sub, "pred")
        actual = weighted_index(sub, "actual")
        err = (pred - actual) / actual * 100
        print(f"\n[Tier {tier_label}] 품목수 {len(sub)}, 가중치합 {sub['가중치'].sum():.1f}")
        print(f"  오차% 범위: {err.min():.2f} ~ {err.max():.2f} / 평균절대오차: {err.abs().mean():.3f}")
        for m in TARGET_MONTHS:
            tier_rows.append({"Tier": tier_label, "월": m, "예측": pred[m], "실제": actual[m], "오차%": err[m]})
    pd.DataFrame(tier_rows).to_csv(SCRIPTS / "all_tiers_tier_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 78)
    print("품목별 정확도 (MAPE) - Tier/모델별 요약")
    print("=" * 78)
    item_stats = []
    for _, row in df.iterrows():
        preds = row[[f"pred_{m}" for m in TARGET_MONTHS]].astype(float).values
        actuals = row[[f"actual_{m}" for m in TARGET_MONTHS]].astype(float).values
        ape = np.abs((preds - actuals) / actuals) * 100
        mape = np.nanmean(ape)
        item_stats.append({"품목명": row["품목명"], "Tier": row["Tier"], "가중치": row["가중치"],
                            "모델": row["모델"], "MAPE%": mape})
    item_df = pd.DataFrame(item_stats)
    item_df.to_csv(SCRIPTS / "all_tiers_item_accuracy.csv", index=False, encoding="utf-8-sig")

    print("\nTier/모델별 가중평균 MAPE:")
    for (tier, model), g in item_df.groupby(["Tier", "모델"]):
        wavg = (g["MAPE%"] * g["가중치"]).sum() / g["가중치"].sum()
        print(f"  Tier {tier} / {model}: 품목수 {len(g)}, 가중평균MAPE {wavg:.3f}%")

    overall_wavg = (item_df["MAPE%"] * item_df["가중치"]).sum() / item_df["가중치"].sum()
    print(f"\n전체 458개 품목 가중평균 MAPE: {overall_wavg:.3f}%")

    print("\n가중치 1.0 이상 중 MAPE 최고 15개:")
    worst = item_df[item_df["가중치"] >= 1.0].sort_values("MAPE%", ascending=False).head(15)
    print(worst[["품목명", "Tier", "모델", "가중치", "MAPE%"]].to_string(index=False))

    print("\n저장 완료:", SCRIPTS / "all_tiers_total_index_comparison.csv")


if __name__ == "__main__":
    main()
