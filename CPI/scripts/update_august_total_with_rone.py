"""
8월 바텀업 총지수 재계산 - 전세·월세를 R-ONE 주간기반 예측치로 교체.

기존 forecast_august_2026.py + apply_bias_correction_august.py 파이프라인은
전세·월세를 자체 히스토리 ETS로만 예측했다(106.326 / 105.947). 이후 R-ONE
데이터(주간 아파트 실측 + 유형별 회귀 + Jevons 합성 + CPI 회귀모델, 세 가지
방법 교차검증으로 좁혀진 값)로 각각 106.26 / 105.92를 별도 산출했다
(8월_전세상승률_예측_R-ONE주간기반.md, 8월_월세상승률_예측_R-ONE주간기반.md).

이 스크립트는 458개 품목 바텀업 집계에서 이 두 품목만 R-ONE 기반 값으로 교체하고
나머지 456개 품목은 그대로 둔 채 총지수를 재계산한다(가중치 54.2+44.9=99.1,
전체 1000.0 중 9.91%).
"""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

RONE_JEONSE = 106.26
RONE_WOLSE = 105.92


def blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def main():
    df = pd.read_csv(SCRIPTS / "august2026_bottomup_items_corrected.csv", encoding="utf-8-sig")
    w = df["가중치"].values

    before_total = blend(df["보정후_pred_2026-08"].values, w)

    jeonse_before = df.loc[df["품목명"] == "전세", "보정후_pred_2026-08"].iloc[0]
    wolse_before = df.loc[df["품목명"] == "월세", "보정후_pred_2026-08"].iloc[0]

    df["보정후_pred_2026-08(R-ONE반영)"] = df["보정후_pred_2026-08"]
    df.loc[df["품목명"] == "전세", "보정후_pred_2026-08(R-ONE반영)"] = RONE_JEONSE
    df.loc[df["품목명"] == "월세", "보정후_pred_2026-08(R-ONE반영)"] = RONE_WOLSE

    after_total = blend(df["보정후_pred_2026-08(R-ONE반영)"].values, w)

    print(f"전세: {jeonse_before:.3f} -> {RONE_JEONSE:.3f} (가중치 {df.loc[df['품목명']=='전세','가중치'].iloc[0]})")
    print(f"월세: {wolse_before:.3f} -> {RONE_WOLSE:.3f} (가중치 {df.loc[df['품목명']=='월세','가중치'].iloc[0]})")
    print()
    print(f"교체 전 바텀업 총지수: {before_total:.4f}")
    print(f"교체 후 바텀업 총지수(R-ONE 반영): {after_total:.4f}")
    print(f"변화: {after_total - before_total:+.4f}")

    df.to_csv(SCRIPTS / "august2026_bottomup_items_roneupdate.csv", index=False, encoding="utf-8-sig")
    return before_total, after_total


if __name__ == "__main__":
    main()
