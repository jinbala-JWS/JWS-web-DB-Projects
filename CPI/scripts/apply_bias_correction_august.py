"""
8월 바텀업 예측(forecast_august_2026.py 결과)에 게이트 기반 트레일링 편향보정을 적용.
(항목별_편향보정_결과.md 3장에서 검증된 방식과 동일한 파라미터: 트레일링 6개월,
방향일치 80%이상, 표준편차 2%p이하, 축소계수 0.6)

트레일링 잔차는 이미 계산해둔 12개월 백테스트(all_tiers_forecast_vs_actual.csv,
2025-08~2026-07의 워크포워드 pred/actual)에서 가장 최근 6개월(2026-02~2026-07)을 사용 —
8월 예측 시점 기준으로 전부 "과거" 데이터라 미래정보 누출 없음.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

TRAILING_MONTHS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]
SHRINKAGE = 0.6
CONSISTENCY_MIN = 0.8
STD_MAX_PCT = 2.0


def blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def main():
    backtest = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv")
    aug = pd.read_csv(SCRIPTS / "august2026_bottomup_items.csv")

    corrections = {}
    for _, row in backtest.iterrows():
        item = row["품목명"]
        lvl = [row[f"pred_{m}"] - row[f"actual_{m}"] for m in TRAILING_MONTHS]
        pct = [(row[f"pred_{m}"] - row[f"actual_{m}"]) / row[f"actual_{m}"] * 100 for m in TRAILING_MONTHS]
        mean_pct = np.mean(pct)
        consistency = np.mean([np.sign(p) == np.sign(mean_pct) for p in pct])
        std_pct = np.std(pct)
        if consistency >= CONSISTENCY_MIN and std_pct <= STD_MAX_PCT and abs(mean_pct) >= 0.05:
            corrections[item] = np.mean(lvl)
        else:
            corrections[item] = 0.0

    aug["보정치(레벨)"] = aug["품목명"].map(corrections).fillna(0.0)
    aug["보정후_pred_2026-08"] = aug["pred_2026-08"] - SHRINKAGE * aug["보정치(레벨)"]

    w = aug["가중치"].values
    raw_total = blend(aug["pred_2026-08"].values, w)
    corr_total = blend(aug["보정후_pred_2026-08"].values, w)

    n_corrected = (aug["보정치(레벨)"] != 0).sum()
    print(f"보정 적용 품목수: {n_corrected} / {len(aug)}")
    print(f"보정 전 바텀업 총지수: {raw_total:.3f}")
    print(f"보정 후 바텀업 총지수: {corr_total:.3f}")

    aug.to_csv(SCRIPTS / "august2026_bottomup_items_corrected.csv", index=False, encoding="utf-8-sig")

    print("\n주요 품목 보정 내역:")
    for name in ["휴대전화기", "사립대학교납입금", "등유", "자동차용LPG", "전세", "월세", "컴퓨터"]:
        r = aug[aug["품목명"] == name]
        if len(r):
            r = r.iloc[0]
            print(f"  {name}: {r['pred_2026-08']:.3f} -> {r['보정후_pred_2026-08']:.3f} "
                  f"(보정치 {r['보정치(레벨)']:+.3f})")

    return raw_total, corr_total


if __name__ == "__main__":
    main()
