"""
9월 바텀업 총지수 재계산 - 전세·월세를 R-ONE 기반 예측치로 교체.

8월과 달리 이번엔 R-ONE "주택종합" 전세·월세 가격지수(한국부동산원 내부 API
sttsDataPreviewList.do, statblId=A_2024_00019/00022)의 8월 실측치를 직접 확보해
1개월 시차 회귀모델(R-ONE_CPI_전월세_회귀모델.md, 이번에 8월 데이터 포인트 추가해
재적합: 전세 slope=0.4179/intercept=-0.0077/R²=0.59, 월세 slope=0.3027/intercept=0.0295/R²=0.50)에
대입해 9월 CPI 전세·월세를 예측했다(106.531 / 106.185).

기존 forecast_september_2026.py는 전세·월세를 자체 히스토리 비계절ETS로 예측했다
(106.558 / 106.169). 두 방법이 0.03%p 이내로 거의 일치해 서로를 뒷받침한다.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

RONE_JEONSE = 106.531
RONE_WOLSE = 106.185


def blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def main():
    df = pd.read_csv(SCRIPTS / "september2026_bottomup_items_corrected.csv", encoding="utf-8-sig")
    w = df["가중치"].values

    before_total = blend(df["보정후_pred_2026-09"].values, w)

    jeonse_before = df.loc[df["품목명"] == "전세", "보정후_pred_2026-09"].iloc[0]
    wolse_before = df.loc[df["품목명"] == "월세", "보정후_pred_2026-09"].iloc[0]

    df["보정후_pred_2026-09(R-ONE반영)"] = df["보정후_pred_2026-09"]
    df.loc[df["품목명"] == "전세", "보정후_pred_2026-09(R-ONE반영)"] = RONE_JEONSE
    df.loc[df["품목명"] == "월세", "보정후_pred_2026-09(R-ONE반영)"] = RONE_WOLSE

    after_total = blend(df["보정후_pred_2026-09(R-ONE반영)"].values, w)

    print(f"전세: {jeonse_before:.3f} -> {RONE_JEONSE:.3f} (가중치 {df.loc[df['품목명']=='전세','가중치'].iloc[0]})")
    print(f"월세: {wolse_before:.3f} -> {RONE_WOLSE:.3f} (가중치 {df.loc[df['품목명']=='월세','가중치'].iloc[0]})")
    print()
    print(f"교체 전 바텀업 총지수: {before_total:.4f}")
    print(f"교체 후 바텀업 총지수(R-ONE 반영): {after_total:.4f}")
    print(f"변화: {after_total - before_total:+.4f}")

    df.to_csv(SCRIPTS / "september2026_bottomup_items_roneupdate.csv", index=False, encoding="utf-8-sig")
    return before_total, after_total


if __name__ == "__main__":
    main()
