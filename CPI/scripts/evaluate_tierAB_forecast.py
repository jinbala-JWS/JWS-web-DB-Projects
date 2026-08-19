"""tierAB_forecast_vs_actual.csv를 가중집계 지수로 합산하고, 실제값과 비교 검증한다."""
import numpy as np
import pandas as pd
from pathlib import Path

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"
TARGET_MONTHS = [f"2026-{m:02d}" for m in range(1, 8)]


def weighted_index(df: pd.DataFrame, col_prefix: str) -> pd.Series:
    out = {}
    wsum = df["가중치"].sum()
    for m in TARGET_MONTHS:
        out[m] = (df[f"{col_prefix}_{m}"] * df["가중치"]).sum() / wsum
    return pd.Series(out)


def main():
    df = pd.read_csv(SCRIPTS / "tierAB_forecast_vs_actual.csv")

    print("=" * 70)
    print("Tier별 가중집계 지수: 예측 vs 실제 (2026-01~07)")
    print("=" * 70)

    agg_rows = []
    for tier_label, sub in [("A", df[df.Tier == "A"]), ("B", df[df.Tier == "B"]), ("A+B", df)]:
        pred = weighted_index(sub, "pred")
        actual = weighted_index(sub, "actual")
        err_pct = (pred - actual) / actual * 100
        print(f"\n[Tier {tier_label}] (품목수 {len(sub)}, 가중치합 {sub['가중치'].sum():.1f})")
        comp = pd.DataFrame({"예측": pred.round(3), "실제": actual.round(3), "오차%": err_pct.round(3)})
        print(comp.to_string())
        for m in TARGET_MONTHS:
            agg_rows.append({
                "Tier": tier_label, "월": m,
                "예측지수": pred[m], "실제지수": actual[m], "오차%": err_pct[m],
            })

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(SCRIPTS / "tierAB_aggregate_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("품목별 예측 정확도 (MAPE, 방향적중률) - 상위/하위 요약")
    print("=" * 70)
    item_stats = []
    for _, row in df.iterrows():
        preds = row[[f"pred_{m}" for m in TARGET_MONTHS]].astype(float).values
        actuals = row[[f"actual_{m}" for m in TARGET_MONTHS]].astype(float).values
        ape = np.abs((preds - actuals) / actuals) * 100
        mape = np.nanmean(ape)

        pred_diff = np.diff(preds)
        actual_diff = np.diff(actuals)
        hit = np.mean(np.sign(pred_diff) == np.sign(actual_diff))
        item_stats.append({
            "품목명": row["품목명"], "Tier": row["Tier"], "가중치": row["가중치"],
            "MAPE%": mape, "방향적중률": hit,
        })
    item_df = pd.DataFrame(item_stats)
    item_df.to_csv(SCRIPTS / "tierAB_item_accuracy.csv", index=False, encoding="utf-8-sig")

    print(f"\n전체 418개 품목 MAPE 분포:\n{item_df['MAPE%'].describe().round(3)}")
    print(f"\n가중평균 MAPE (weight-weighted): {(item_df['MAPE%']*item_df['가중치']).sum()/item_df['가중치'].sum():.3f}%")
    print(f"전체 평균 방향적중률: {item_df['방향적중률'].mean():.3f}")

    print("\nMAPE 가장 낮은(정확한) 10개 품목:")
    print(item_df.sort_values("MAPE%").head(10)[["품목명", "Tier", "가중치", "MAPE%"]].to_string(index=False))

    print("\nMAPE 가장 높은(부정확한) 10개 품목 (가중치 0.5 이상만):")
    worst = item_df[item_df["가중치"] >= 0.5].sort_values("MAPE%", ascending=False).head(10)
    print(worst[["품목명", "Tier", "가중치", "MAPE%"]].to_string(index=False))

    print("\n저장:", SCRIPTS / "tierAB_aggregate_comparison.csv", "/", SCRIPTS / "tierAB_item_accuracy.csv")


if __name__ == "__main__":
    main()
