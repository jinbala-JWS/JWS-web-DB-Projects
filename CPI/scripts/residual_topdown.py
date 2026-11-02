"""
탑다운 채택 카테고리 중, R-ONE처럼 "외부 실측 데이터를 끌어올 수 있는" 세부항목이
있는 경우 그 항목만 바텀업(실측 기반)으로 빼고, 나머지("remainder")만의 변동률로
탑다운을 적용하는 모듈. 2026-10-20 대화에서 논의된 방식을 2026-11-01에 코드화했다.

기존 방식: 카테고리(예: "04 주택,수도,전기및연료") 전체 히스토리에 계절ETS를 직접
적합해 탑다운 예측치를 뽑았다 — 이때 전세·월세처럼 R-ONE 선행지표로 꽤 정확하게
예측 가능한 항목의 정보가 카테고리 전체 추세에 희석되어 버린다.

새 방식:
1. 카테고리 히스토리(official, 가중치 W_cat)에서 pullable 항목들(가중치 w_i, 자체
   실측 히스토리 series_i)의 기여분을 제거해 "remainder" 시계열을 역산한다:
       remainder(t) = (cat(t)*W_cat - sum_i(w_i * series_i(t))) / (W_cat - sum_i(w_i))
2. remainder 시계열 자체에 계절ETS(탑다운과 동일한 모델)를 적합해 예측한다.
3. pullable 항목은 각자의 (R-ONE 등 실측 기반) 예측치를 쓰고, remainder 예측치와
   가중평균으로 재조합해 카테고리 최종예측을 만든다.

현재 pullable로 검증된 항목은 "04 주택,수도,전기및연료" 안의 전세·월세뿐이다
(R-ONE_CPI_전월세_회귀모델.md, 3개월 연속 바텀업 ETS와 0.02%p 이내로 교차검증됨).
다른 카테고리(06 보건·09 오락및문화)는 아직 이런 실측 pull 가능 항목이 없어 그대로
카테고리 전체 탑다운을 유지한다.
"""
import pandas as pd

# {카테고리명(official 라벨과 동일): {품목명: 가중치}}
PULLABLE_ITEMS = {
    "04 주택, 수도, 전기 및 연료": {"전세": 54.2, "월세": 44.9},
}


def build_remainder_series(cat_series: pd.Series, cat_weight: float,
                            pulled_series: dict, pulled_weights: dict) -> pd.Series:
    """카테고리 히스토리에서 pulled 항목 기여분을 제거한 remainder 시계열(가중치 W_rem 기준 지수)."""
    w_pulled_sum = sum(pulled_weights.values())
    w_rem = cat_weight - w_pulled_sum
    acc = cat_series * cat_weight
    for name, w in pulled_weights.items():
        acc = acc - w * pulled_series[name]
    return acc / w_rem, w_rem


def residual_topdown_for_category(category_label: str, official: pd.DataFrame, item_col: str,
                                    month_cols_sorted: list, target: str, cat_weight: float,
                                    panel_series: dict, pulled_forecasts: dict,
                                    ets_forecast_fn) -> dict:
    """
    category_label의 카테고리 전체 탑다운 대신 remainder-탑다운 + pulled 바텀업 재조합을 수행.

    panel_series: {품목명: pd.Series(월 index -> 실측 지수, 전체 히스토리)} - pullable 항목용
    pulled_forecasts: {품목명: target월 예측치(실측기반, 예: R-ONE 보정 바텀업)}
    ets_forecast_fn: forecast_september_2026.ets_forecast 같은 (hist, is_seasonal) -> pred 함수

    반환: {"탑다운예측": 재조합된 카테고리 최종값, "remainder예측": remainder 자체 예측,
          "remainder가중치": w_rem}
    """
    pulled_weights = PULLABLE_ITEMS[category_label]
    cat_row = official[official[item_col] == category_label].iloc[0]
    cat_series = pd.Series({c: float(cat_row[c]) for c in month_cols_sorted})

    remainder_series, w_rem = build_remainder_series(
        cat_series, cat_weight,
        {name: panel_series[name] for name in pulled_weights}, pulled_weights,
    )

    train_cols = [c for c in month_cols_sorted if c < target]
    hist_rem = remainder_series[train_cols].dropna()
    pred_rem = ets_forecast_fn(hist_rem, is_seasonal=True)

    combined = pred_rem * w_rem
    for name, w in pulled_weights.items():
        combined += w * pulled_forecasts[name]
    combined /= cat_weight

    return {"탑다운예측": combined, "remainder예측": pred_rem, "remainder가중치": w_rem}
