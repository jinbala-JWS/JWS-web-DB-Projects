"""
Tier A(일반 시장가격형)/B(계절형) 418개 품목 — 2025년까지 데이터로 2026년 1월을 예측하고,
그 뒤로는 매달 실제데이터가 들어왔다고 가정한 롤링(워크포워드) 1개월 예측을 2026년 7월까지 반복한다.
각 스텝마다 그 시점까지의 실제 데이터로 모델을 다시 적합(refit)한다.

모델:
- Tier A: ETS(추세=완만감쇠, 계절성 없음) — "일반 시장가격형"은 계절성 가정하지 않는다는 설계 원칙.
- Tier B(일반 계절): ETS(추세=완만감쇠, 계절성=월별)
- Tier B(고정계절 11개 품목): 2017-01 이후 데이터만 사용(이월->대체 방식 변경 이전 데이터는 배제),
  동일하게 계절 ETS 적용.
- 데이터가 너무 짧거나(< 24개월) 모델 적합이 실패하면 단순 계절나이브/랜덤워크로 대체.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"

TARGET_MONTHS = [f"2026-{m:02d}" for m in range(1, 8)]  # 2026-01 ~ 2026-07
FIXED_SEASON_NAMES = {
    "복숭아", "포도", "감", "귤", "오렌지", "참외", "수박", "딸기", "체리", "열무", "굴",
}


def fit_forecast_one(hist: pd.Series, is_seasonal: bool, restrict_2017: bool) -> float:
    """hist: 시간순 정렬된 결측없는 시리즈(마지막 값이 '이번에 아는 마지막 실측월'). 다음 1개월 예측."""
    s = hist.copy()
    if restrict_2017:
        s = s[s.index >= "2017-01"]
    s = s.dropna()

    if len(s) < 24:
        # 데이터 부족 -> 최근 12개월 평균 변화율을 반영한 단순 랜덤워크
        if len(s) >= 2:
            drift = s.diff().dropna().tail(12).mean()
            return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))
        return float(s.iloc[-1]) if len(s) else np.nan

    try:
        if is_seasonal and len(s) >= 36:
            model = ExponentialSmoothing(
                s.values, trend="add", damped_trend=True,
                seasonal="add", seasonal_periods=12,
                initialization_method="estimated",
            )
        else:
            model = ExponentialSmoothing(
                s.values, trend="add", damped_trend=True, seasonal=None,
                initialization_method="estimated",
            )
        fit = model.fit(optimized=True)
        pred = fit.forecast(1)[0]
        return float(pred)
    except Exception:
        drift = s.diff().dropna().tail(12).mean()
        return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))


def main():
    panel = pd.read_csv(SCRIPTS / "tierAB_monthly_panel.csv")
    month_cols = [c for c in panel.columns if c[:2] in ("19", "20")]
    month_cols_sorted = sorted(month_cols)

    results = []
    n = len(panel)
    for i, row in panel.iterrows():
        item = row["품목명"]
        tier = row["Tier"]
        weight = row["가중치"]
        is_seasonal = tier == "B"
        restrict_2017 = item in FIXED_SEASON_NAMES

        full_series = pd.Series(
            row[month_cols_sorted].astype(float).values, index=month_cols_sorted
        )
        # 결측 보간(내부 결측만) - 시작 이전 결측은 그대로 잘라냄
        full_series = full_series.interpolate(limit_area="inside")

        preds = {}
        for target in TARGET_MONTHS:
            # target 이전 달까지의 "실제" 데이터만 사용 (워크포워드)
            train_end_idx = month_cols_sorted.index(target) - 1
            train_cols = month_cols_sorted[: train_end_idx + 1]
            hist = full_series[train_cols].dropna()
            if len(hist) == 0:
                preds[target] = np.nan
                continue
            pred = fit_forecast_one(hist, is_seasonal=is_seasonal, restrict_2017=restrict_2017)
            preds[target] = pred

        rec = {
            "품목코드": row["품목코드"], "품목명": item, "Tier": tier, "가중치": weight,
        }
        for target in TARGET_MONTHS:
            rec[f"pred_{target}"] = preds[target]
            rec[f"actual_{target}"] = full_series.get(target, np.nan)
        results.append(rec)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} 품목 처리 완료 ({item})")

    out = pd.DataFrame(results)
    out.to_csv(SCRIPTS / "tierAB_forecast_vs_actual.csv", index=False, encoding="utf-8-sig")
    print("저장:", SCRIPTS / "tierAB_forecast_vs_actual.csv", out.shape)


if __name__ == "__main__":
    main()
