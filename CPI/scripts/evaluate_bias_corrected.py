"""
항목별 트레일링 편향보정(rolling bias correction)을 적용해 검증구간(2025-08~2026-07)을 재평가.

방식: 각 검증월 M에 대해, 그 항목의 "M보다 이전"인 최근 최대 6개월의 실제 예측오차
(pred-actual, 레벨 기준)의 평균을 보정계수로 삼아 0.6배(축소계수, 과도한 보정 방지)만큼
원래 예측값에서 빼준다. 번인구간(2024-08~2025-07)이 있어 검증구간 첫 달부터도
충분한 트레일링 잔차를 확보할 수 있음 - 미래정보 누출 없음(항상 그 시점 "이전" 잔차만 사용).
"""
import numpy as np
import pandas as pd
from pathlib import Path

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"

BURNIN_MONTHS = [
    "2024-08", "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07",
]
TEST_MONTHS = [
    "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
]
ALL_MONTHS = BURNIN_MONTHS + TEST_MONTHS
TRAILING_WINDOW = 6
SHRINKAGE = 0.6


def weighted_index(vals, w, blend=True):
    arith = np.average(vals, weights=w)
    if not blend:
        return arith
    geom = np.exp(np.average(np.log(vals), weights=w))
    return 0.5 * arith + 0.5 * geom


def main():
    df = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv")

    # 게이트 기준: 트레일링 구간의 % 오차가 (1) 부호가 일관되고(80% 이상 동일방향)
    # (2) 상대적으로 안정적(표준편차가 크지 않음)일 때만 보정 적용.
    # 참외·토마토처럼 표준편차 5~18%인 노이즈성 품목에는 보정을 걸지 않는다.
    CONSISTENCY_MIN = 0.8
    STD_MAX_PCT = 2.0  # 트레일링 % 오차의 표준편차가 이보다 크면 보정 안 함(노이즈로 판단)

    corrected_rows = []
    gate_count = 0
    for _, row in df.iterrows():
        preds = {m: row[f"pred_{m}"] for m in ALL_MONTHS}
        actuals = {m: row[f"actual_{m}"] for m in ALL_MONTHS}
        residuals = {m: preds[m] - actuals[m] for m in ALL_MONTHS}
        pct_residuals = {m: (preds[m] - actuals[m]) / actuals[m] * 100 for m in ALL_MONTHS}

        rec = {"품목코드": row["품목코드"], "품목명": row["품목명"], "Tier": row["Tier"],
               "가중치": row["가중치"], "모델": row["모델"]}
        for m in TEST_MONTHS:
            idx = ALL_MONTHS.index(m)
            trailing_lvl = [residuals[ALL_MONTHS[j]] for j in range(max(0, idx - TRAILING_WINDOW), idx)]
            trailing_pct = [pct_residuals[ALL_MONTHS[j]] for j in range(max(0, idx - TRAILING_WINDOW), idx)]
            trailing_lvl = [t for t in trailing_lvl if pd.notna(t)]
            trailing_pct = [t for t in trailing_pct if pd.notna(t)]

            correction = 0.0
            if len(trailing_pct) >= 3:
                mean_pct = np.mean(trailing_pct)
                consistency = np.mean([np.sign(t) == np.sign(mean_pct) for t in trailing_pct])
                std_pct = np.std(trailing_pct)
                if consistency >= CONSISTENCY_MIN and std_pct <= STD_MAX_PCT and abs(mean_pct) >= 0.05:
                    correction = np.mean(trailing_lvl)
                    gate_count += 1

            corrected_pred = preds[m] - SHRINKAGE * correction
            rec[f"pred_{m}"] = corrected_pred
            rec[f"actual_{m}"] = actuals[m]
            rec[f"raw_pred_{m}"] = preds[m]
        corrected_rows.append(rec)
    print(f"보정이 실제로 적용된 (품목,월) 조합: {gate_count} / {len(df)*len(TEST_MONTHS)}")

    cdf = pd.DataFrame(corrected_rows)
    cdf.to_csv(SCRIPTS / "all_tiers_forecast_bias_corrected.csv", index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f"항목별 트레일링 편향보정(window={TRAILING_WINDOW}개월, shrinkage={SHRINKAGE}) 적용 후")
    print("=" * 78)

    official = pd.read_csv(SCRIPTS / "official_total_index.csv")
    official_vals = official[official.iloc[:, 0] == "0 총지수"][TEST_MONTHS].iloc[0].astype(float)

    w = cdf["가중치"].values
    rows = []
    for m in TEST_MONTHS:
        raw_pred_total = weighted_index(df[f"pred_{m}"].astype(float).values, w)
        corr_pred_total = weighted_index(cdf[f"pred_{m}"].astype(float).values, w)
        actual_total = official_vals[m]
        rows.append({
            "월": m,
            "보정전_예측": raw_pred_total, "보정후_예측": corr_pred_total, "실제(공식)": actual_total,
            "오차%_보정전": (raw_pred_total - actual_total) / actual_total * 100,
            "오차%_보정후": (corr_pred_total - actual_total) / actual_total * 100,
        })
    comp = pd.DataFrame(rows).set_index("월")
    print(comp.round(3).to_string())
    comp.to_csv(SCRIPTS / "bias_correction_comparison.csv", encoding="utf-8-sig")

    mae_before = comp["오차%_보정전"].abs().mean()
    mae_after = comp["오차%_보정후"].abs().mean()
    mean_before = comp["오차%_보정전"].mean()
    mean_after = comp["오차%_보정후"].mean()
    print(f"\n평균오차(부호포함): 보정전 {mean_before:.3f}%p -> 보정후 {mean_after:.3f}%p")
    print(f"평균절대오차(MAE):   보정전 {mae_before:.3f}%p -> 보정후 {mae_after:.3f}%p")

    # 품목별 MAPE 개선 여부
    item_rows = []
    for i in range(len(df)):
        raw_p = df.iloc[i][[f"pred_{m}" for m in TEST_MONTHS]].astype(float).values
        corr_p = cdf.iloc[i][[f"pred_{m}" for m in TEST_MONTHS]].astype(float).values
        act = df.iloc[i][[f"actual_{m}" for m in TEST_MONTHS]].astype(float).values
        mape_raw = np.mean(np.abs((raw_p - act) / act)) * 100
        mape_corr = np.mean(np.abs((corr_p - act) / act)) * 100
        item_rows.append({
            "품목명": df.iloc[i]["품목명"], "Tier": df.iloc[i]["Tier"], "가중치": df.iloc[i]["가중치"],
            "MAPE_보정전": mape_raw, "MAPE_보정후": mape_corr, "개선": mape_raw - mape_corr,
        })
    item_comp = pd.DataFrame(item_rows)
    item_comp.to_csv(SCRIPTS / "bias_correction_item_comparison.csv", index=False, encoding="utf-8-sig")

    wavg_before = (item_comp["MAPE_보정전"] * item_comp["가중치"]).sum() / item_comp["가중치"].sum()
    wavg_after = (item_comp["MAPE_보정후"] * item_comp["가중치"]).sum() / item_comp["가중치"].sum()
    print(f"\n전체 458개 품목 가중평균 MAPE: 보정전 {wavg_before:.3f}% -> 보정후 {wavg_after:.3f}%")

    n_improved = (item_comp["개선"] > 0.01).sum()
    n_worsened = (item_comp["개선"] < -0.01).sum()
    print(f"개선된 품목: {n_improved}개 / 악화된 품목: {n_worsened}개 / 변화없음: {len(item_comp)-n_improved-n_worsened}개")

    print("\n가중치 1.0 이상 중 가장 많이 개선된 품목 10개:")
    top_improved = item_comp[item_comp["가중치"] >= 1.0].sort_values("개선", ascending=False).head(10)
    print(top_improved.to_string(index=False))

    print("\n가중치 1.0 이상 중 가장 많이 악화된 품목 10개:")
    top_worsened = item_comp[item_comp["가중치"] >= 1.0].sort_values("개선").head(10)
    print(top_worsened.to_string(index=False))


if __name__ == "__main__":
    main()
