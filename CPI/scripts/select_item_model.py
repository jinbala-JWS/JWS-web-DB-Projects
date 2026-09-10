"""
아직 특별 처리 없이 기본 비계절ETS를 쓰고 있는 품목들(Tier B 제외, 이미 검증된
품목 제외) 전체를 대상으로, 4가지 예측방식을 24개월 워크포워드로 비교해 품목별로
가장 정확한 방식을 자동 선택한다.

후보 방식:
  1) 비계절ETS      - 현재 기본값(Holt 감쇠추세)
  2) 계절ETS        - 12개월 계절성 포함
  3) 계단평균(최근5회 가중) - 예측 대상월과 같은 "달"로의 과거 전월대비 변동만 모아
     최근 5회 관측치를 가중평균(최신에 가중치 크게: 5,4,3,2,1)한 값을 그 달의
     기대 변동률로 쓰고, 그 외 달은 0%(무변동)로 둔다 - "그 달에만 계단식으로
     오르고 나머진 안 움직인다"는 패턴을 정면으로 모델링.
  4) 완전동결(0%)   - 그냥 직전 실측치를 그대로 이월(무조건 무변동 예측)

품목별로 4개 중 MAPE가 가장 낮은 방식을 채택하되, **현재 기본(비계절ETS) 대비
15% 이상 상대개선**이 없으면 굳이 바꾸지 않는다(과최적화 방지 - 작은 표본 노이즈로
방식이 바뀌는 것을 막는 최소 개선폭 게이트).
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

SCRIPTS = Path(__file__).resolve().parent
MIN_WEIGHT = 1.0
MIN_RELATIVE_IMPROVEMENT = 0.15  # 15% 이상 개선 안 되면 현재 방식 유지

RESOLVED = {
    "외래진료비", "한방진료비", "약국조제료", "치과진료비", "입원진료비",
    "전문대학납입금", "국공립대학교납입금", "사립대학교납입금",
    "국공립대학원납입금", "사립대학원납입금", "요양시설이용료", "전기료",
    "휘발유", "경유", "등유", "자동차용LPG", "취사용LPG",
    "여자외의", "점퍼", "여자하의", "남자상의", "남자하의", "쓰레기봉투료",
    "유아동복", "청바지",
}
TESTED_REJECTED = {
    "휴대전화기", "유치원납입금", "스웨터", "대입전형료", "등산복",
    "보청기", "이러닝이용료", "티셔츠", "여자상의",
}


def ets_forecast(hist: pd.Series, is_seasonal: bool) -> float:
    s = hist.dropna()
    if len(s) < 24:
        drift = s.diff().dropna().tail(12).mean()
        return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))
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


def step_weighted_forecast(hist: pd.Series, target_month_num: int, weights=(5, 4, 3, 2, 1)) -> float:
    """target_month_num(1~12)로의 과거 전월대비 변동만 모아 최근 N회 가중평균."""
    s = hist.dropna()
    pct = s.pct_change().dropna() * 100
    same_month = pct[pd.Index(pct.index).map(lambda x: int(x[5:7])) == target_month_num]
    if len(same_month) == 0:
        return float(s.iloc[-1])
    recent = same_month.tail(len(weights))
    w = np.array(weights[-len(recent):], dtype=float)
    avg_pct = float(np.average(recent.values, weights=w))
    return float(s.iloc[-1] * (1 + avg_pct / 100))


def flat_forecast(hist: pd.Series) -> float:
    return float(hist.dropna().iloc[-1])


def evaluate_item(series: pd.Series, month_cols, test_months):
    rows_err = {"비계절ETS": [], "계절ETS": [], "계단평균": [], "완전동결": []}
    for m in test_months:
        train_cols = [c for c in month_cols if c < m]
        hist = series[train_cols]
        actual = series.get(m, np.nan)
        if pd.isna(actual):
            continue
        target_month_num = int(m[5:7])
        preds = {
            "비계절ETS": ets_forecast(hist, False),
            "계절ETS": ets_forecast(hist, True),
            "계단평균": step_weighted_forecast(hist, target_month_num),
            "완전동결": flat_forecast(hist),
        }
        for k, p in preds.items():
            rows_err[k].append(abs((p - actual) / actual * 100))
    return {k: np.mean(v) for k, v in rows_err.items() if v}


def main():
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv", encoding="utf-8-sig")
    panel.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"] + list(panel.columns[6:])
    month_cols = sorted([c for c in panel.columns if c[:2] in ("19", "20")])
    test_months = [c for c in month_cols if "2024-08" <= c <= "2026-08"]

    target = panel[(panel["Tier"] != "B") & (~panel["품목명"].isin(RESOLVED)) &
                    (~panel["품목명"].isin(TESTED_REJECTED)) & (panel["가중치"] >= MIN_WEIGHT)]
    print(f"대상 품목수: {len(target)} (가중치합 {target['가중치'].sum():.1f})")

    results = []
    for i, (_, r) in enumerate(target.iterrows()):
        series = pd.Series(r[month_cols].astype(float).values, index=month_cols).interpolate(limit_area="inside")
        mapes = evaluate_item(series, month_cols, test_months)
        if not mapes:
            continue
        best_method = min(mapes, key=mapes.get)
        baseline = mapes.get("비계절ETS", np.nan)
        best_val = mapes[best_method]
        rel_improve = (baseline - best_val) / baseline if baseline else 0
        final_method = best_method if (best_method != "비계절ETS" and rel_improve >= MIN_RELATIVE_IMPROVEMENT) else "비계절ETS"
        results.append({
            "품목코드": r["품목코드"], "품목명": r["품목명"], "Tier": r["Tier"], "가중치": r["가중치"],
            **{f"MAPE_{k}": v for k, v in mapes.items()},
            "최선방식": best_method, "상대개선": rel_improve, "채택방식": final_method,
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(target)} 처리...")

    out = pd.DataFrame(results)
    out.to_csv(SCRIPTS / "item_model_selection.csv", index=False, encoding="utf-8-sig")
    print("\n채택방식 분포:")
    print(out["채택방식"].value_counts())
    print("\n채택방식별 가중치합:")
    print(out.groupby("채택방식")["가중치"].sum())
    return out


if __name__ == "__main__":
    main()
