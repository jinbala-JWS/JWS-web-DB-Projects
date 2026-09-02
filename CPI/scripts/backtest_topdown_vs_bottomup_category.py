"""
대분류(12) 레벨에서 탑다운(카테고리 자체 히스토리 계절ETS) vs 바텀업(458개 leaf
예측을 카테고리로 집계) 중 과거에도 어느 쪽이 더 정확했는지 24개월(2024-08~2026-07)
워크포워드로 검증한다.

- 바텀업 leaf별 pred/actual: all_tiers_forecast_vs_actual.csv (이미 계산된 12개월×2
  워크포워드 백테스트 결과, forecast_all_tiers.py/evaluate_all_tiers.py 파이프라인 산출물)
- 탑다운: forecast_august_2026.py의 top_down() 로직(카테고리 자체 히스토리 계절ETS)을
  동일 파라미터로 24개월 각 시점에 대해 재현 - 매 시점 해당 월 이전 데이터만 사용(워크포워드).
- 실제값: cpi_official_monthly_wide.csv의 대분류 행(공식 KOSIS 값).
"""
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

SCRIPTS = Path(__file__).resolve().parent

LETTER_TO_CAT = {
    "A": "01 식료품 및 비주류음료", "B": "02 주류 및 담배", "C": "03 의류 및 신발",
    "D": "04 주택, 수도, 전기 및 연료", "E": "05 가정용품 및 가사 서비스", "F": "06 보건",
    "G": "07 교통", "H": "08 통신", "I": "09 오락 및 문화", "J": "10 교육",
    "K": "11 음식 및 숙박", "L": "12 기타 상품 및 서비스",
}


def ets_forecast(hist: pd.Series, is_seasonal: bool) -> float:
    s = hist.dropna()
    if len(s) < 24:
        if len(s) >= 2:
            drift = s.diff().dropna().tail(12).mean()
            return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))
        return float(s.iloc[-1]) if len(s) else np.nan
    try:
        if is_seasonal and len(s) >= 36:
            model = ExponentialSmoothing(s.values, trend="add", damped_trend=True,
                                          seasonal="add", seasonal_periods=12,
                                          initialization_method="estimated")
        else:
            model = ExponentialSmoothing(s.values, trend="add", damped_trend=True,
                                          seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        drift = s.diff().dropna().tail(12).mean()
        return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))


def blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def run():
    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = official.columns[0]
    official = official.drop_duplicates(subset=item_col, keep="first")
    month_cols_sorted = sorted([c for c in official.columns if c[:2] in ("19", "20")])
    major_rows = official[official[item_col].str.match(r"^\d\d ")]

    bu = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv", encoding="utf-8-sig")
    bu_cols = bu.columns.tolist()
    name_col, tier_col, weight_col = bu_cols[1], bu_cols[2], bu_cols[3]
    code_col = bu_cols[0]
    bu["대분류"] = bu[code_col].str[0].map(LETTER_TO_CAT)

    test_months = sorted([c.replace("pred_", "") for c in bu_cols if c.startswith("pred_")])
    print(f"백테스트 대상 월: {test_months[0]} ~ {test_months[-1]} ({len(test_months)}개월)")

    records = []
    for m in test_months:
        train_cols = [c for c in month_cols_sorted if c < m]
        # 탑다운: 카테고리 자체 히스토리 계절ETS
        for _, r in major_rows.iterrows():
            cat = r[item_col]
            series = pd.Series({c: r[c] for c in month_cols_sorted}).astype(float)
            hist = series[train_cols].dropna()
            actual = series.get(m, np.nan)
            if pd.isna(actual) or len(hist) < 36:
                continue
            pred_td = ets_forecast(hist, is_seasonal=True)
            records.append({"월": m, "분류": cat, "방식": "탑다운", "예측": pred_td, "실제": actual})

        # 바텀업: 해당 월의 458개 leaf pred를 대분류로 집계
        pred_col, actual_col = f"pred_{m}", f"actual_{m}"
        sub = bu[[weight_col, "대분류", pred_col]].dropna()
        for cat, g in sub.groupby("대분류"):
            pred_bu = blend(g[pred_col].values, g[weight_col].values)
            actual_row = major_rows.loc[major_rows[item_col] == cat, m]
            if len(actual_row) == 0 or pd.isna(actual_row.iloc[0]):
                continue
            records.append({"월": m, "분류": cat, "방식": "바텀업", "예측": pred_bu, "실제": actual_row.iloc[0]})

    out = pd.DataFrame(records)
    out["오차"] = out["예측"] - out["실제"]
    out["오차%"] = out["오차"] / out["실제"] * 100
    out.to_csv(SCRIPTS / "topdown_vs_bottomup_category_backtest.csv", index=False, encoding="utf-8-sig")

    summary = out.groupby(["분류", "방식"]).agg(
        MAE=("오차", lambda x: x.abs().mean()),
        MAPE=("오차%", lambda x: x.abs().mean()),
        n=("오차", "count"),
    ).reset_index()
    summary_wide = summary.pivot(index="분류", columns="방식", values=["MAE", "MAPE", "n"])
    summary_wide.to_csv(SCRIPTS / "topdown_vs_bottomup_category_summary.csv", encoding="utf-8-sig")

    print(summary.to_string(index=False))
    return out, summary


if __name__ == "__main__":
    run()
