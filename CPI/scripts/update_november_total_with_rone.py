"""
11월 바텀업 총지수 재계산(교차검증용) - 전세·월세를 R-ONE 기반 예측치로 교체.

R-ONE "주택종합"(전체 주택유형 통합, 전국) 전세·월세 가격지수의 2026-10 실측치를
한국부동산원 내부 API(searchRegionChartData.do, itmId=1000010 전국, BLD_GB 02=전세/03=월세,
hdnChgType=01 전월대비%, dtacycleCd=MM)로 직접 확보해(2026-11-25), 10월 포인트를 추가해
재적합한 1개월 시차 회귀모델(R-ONE_CPI_전월세_회귀모델.md)에 대입했다.

전세: slope=0.4287 intercept=-0.0093 R²=0.6284 (n=14)
월세: slope=0.3155 intercept=0.0273 R²=0.5618 (n=14)
10월 R-ONE(전세 0.34%/월세 0.40%) 입력 -> 11월 CPI 전세 106.836 / 월세 106.503 예측.

기존 forecast_november_2026.py(바텀업 자체 ETS, 11월 MTD 20일 기준)와 비교해 교차검증한다.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

RONE_JEONSE = 106.836
RONE_WOLSE = 106.503


def blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def main():
    df = pd.read_csv(SCRIPTS / "november2026_bottomup_items.csv", encoding="utf-8-sig")
    w = df["가중치"].values

    before_total = blend(df["pred_corr"].values, w)

    jeonse_before = df.loc[df["품목명"] == "전세", "pred_corr"].iloc[0]
    wolse_before = df.loc[df["품목명"] == "월세", "pred_corr"].iloc[0]

    df["pred_corr_roneupdate"] = df["pred_corr"]
    df.loc[df["품목명"] == "전세", "pred_corr_roneupdate"] = RONE_JEONSE
    df.loc[df["품목명"] == "월세", "pred_corr_roneupdate"] = RONE_WOLSE

    after_total = blend(df["pred_corr_roneupdate"].values, w)

    print(f"전세: {jeonse_before:.3f} -> {RONE_JEONSE:.3f} (가중치 {df.loc[df['품목명']=='전세','가중치'].iloc[0]})")
    print(f"월세: {wolse_before:.3f} -> {RONE_WOLSE:.3f} (가중치 {df.loc[df['품목명']=='월세','가중치'].iloc[0]})")
    print()
    print(f"교체 전 바텀업 총지수: {before_total:.4f}")
    print(f"교체 후 바텀업 총지수(R-ONE 반영): {after_total:.4f}")
    print(f"변화: {after_total - before_total:+.4f}")

    df.to_csv(SCRIPTS / "november2026_bottomup_items_roneupdate.csv", index=False, encoding="utf-8-sig")
    return before_total, after_total


if __name__ == "__main__":
    main()
