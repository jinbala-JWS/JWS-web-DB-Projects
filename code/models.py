"""
Tier별 다음달 예측 모델

Tier A (일반 시장가격형): 자체 히스토리로 계절나이브/드리프트/Holt-Winters 중 백테스트 최우수 모델 선택
                        + 상위(대분류) 최근 추세로 축소(shrinkage)
Tier B (계절형): STL 분해(추세+계절)로 예측. 고정계절 11개 품목은 미출하월에는
                통계청 공식 "대체" 로직(같은 신선분류군 비계절품목 평균 모멘텀 적용)
Tier C (정기개정형): 기본값 = 보합(전월과 동일). manual_overrides.json으로 수동 반영 가능
Tier D (원자재연동형): 오피넷 실측이 있는 품목은 외부지표 회귀, 없는 품목은 자체모델/보합 폴백
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

CODE_DIR = Path(__file__).resolve().parent
OVERRIDES_PATH = CODE_DIR / "manual_overrides.json"

# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------


def load_overrides() -> dict:
    if OVERRIDES_PATH.exists():
        return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    return {}


def mom_series(s: pd.Series) -> pd.Series:
    return s.pct_change() * 100


def next_month_str(last_month: str) -> str:
    y, m = int(last_month[:4]), int(last_month[5:7])
    if m == 12:
        return f"{y+1}-01"
    return f"{y}-{m+1:02d}"


def target_month_number(ym: str) -> int:
    return int(ym[5:7])


# ---------------------------------------------------------------------------
# Tier A: 일반 시장가격형
# ---------------------------------------------------------------------------


def _candidate_seasonal_naive(hist: pd.Series) -> float | None:
    if len(hist) < 13:
        return None
    mom = hist.iloc[-1] / hist.iloc[-13] - 1  # 작년 동월 대비 12개월 누적 변동을 그대로 반복한다고 가정하기보다
    # 더 안정적인 방식: 작년 이맘때의 "그 다음달로 넘어가는 월간 변동률"을 그대로 적용
    if len(hist) < 14:
        return None
    last_year_mom = hist.iloc[-12] / hist.iloc[-13] - 1
    return hist.iloc[-1] * (1 + last_year_mom)


def _candidate_drift(hist: pd.Series, n: int = 6) -> float | None:
    if len(hist) < n + 1:
        return None
    recent = hist.iloc[-(n + 1):]
    moms = recent.pct_change().dropna()
    return hist.iloc[-1] * (1 + moms.mean())


def _candidate_holt_winters(hist: pd.Series) -> float | None:
    if len(hist) < 25:
        return None
    try:
        model = ExponentialSmoothing(
            hist.values, trend="add", seasonal="add", seasonal_periods=12,
            initialization_method="estimated",
        ).fit(optimized=True)
        fc = model.forecast(1)[0]
        return float(fc)
    except Exception:
        return None


def _backtest_pick_best(hist: pd.Series, n_test: int = 6) -> str:
    """최근 n_test개월을 1-step 백테스트해서 MAE가 가장 낮은 후보 방식을 고른다."""
    if len(hist) < 20:
        return "drift"
    errors = {"seasonal_naive": [], "drift": [], "holt_winters": []}
    for i in range(len(hist) - n_test, len(hist)):
        train = hist.iloc[:i]
        actual = hist.iloc[i]
        if len(train) < 14:
            continue
        preds = {
            "seasonal_naive": _candidate_seasonal_naive(train),
            "drift": _candidate_drift(train),
            "holt_winters": _candidate_holt_winters(train),
        }
        for k, v in preds.items():
            if v is not None and np.isfinite(v):
                errors[k].append(abs(v - actual))
    avg_err = {k: (np.mean(v) if v else np.inf) for k, v in errors.items()}
    return min(avg_err, key=avg_err.get)


def forecast_tier_a(hist: pd.Series, parent_trend_mom: float | None = None,
                     shrinkage: float = 0.25) -> dict:
    """개별 품목(또는 소분류) 시계열 -> 다음달 예측치.
    parent_trend_mom: 상위분류(대분류)의 최근 3개월 평균 MoM%(0~100 스케일 아님, 소수, 예 0.005=0.5%)
    shrinkage: 상위분류 추세 쪽으로 끌어당기는 비중(변동성 큰 품목일수록 크게 적용)"""
    hist = hist.dropna()
    if len(hist) < 3:
        return {"forecast": hist.iloc[-1] if len(hist) else np.nan, "method": "insufficient_data"}

    method = _backtest_pick_best(hist)
    candidates = {
        "seasonal_naive": _candidate_seasonal_naive(hist),
        "drift": _candidate_drift(hist),
        "holt_winters": _candidate_holt_winters(hist),
    }
    pred = candidates.get(method)
    if pred is None or not np.isfinite(pred):
        pred = _candidate_drift(hist) or hist.iloc[-1]
        method = "drift_fallback"

    last_val = hist.iloc[-1]
    item_mom = pred / last_val - 1

    # 노이즈가 큰(최근 12개월 MoM% 표준편차가 큰) 품목은 상위분류 추세로 더 많이 축소
    recent_mom = hist.pct_change().dropna().iloc[-12:]
    vol = recent_mom.std() if len(recent_mom) >= 6 else 0.0
    dyn_shrink = min(0.6, shrinkage + vol * 3) if parent_trend_mom is not None else 0.0

    if parent_trend_mom is not None and np.isfinite(parent_trend_mom):
        blended_mom = (1 - dyn_shrink) * item_mom + dyn_shrink * parent_trend_mom
    else:
        blended_mom = item_mom

    forecast_val = last_val * (1 + blended_mom)
    return {"forecast": forecast_val, "method": method, "mom_pct": blended_mom * 100,
            "shrinkage_applied": dyn_shrink}


# ---------------------------------------------------------------------------
# Tier B: 계절형
# ---------------------------------------------------------------------------

# 세부 컬럼에서 "미출하" 구간을 파싱하기 위한 월 이름 -> 숫자
_MONTH_MAP = {f"{m}월": m for m in range(1, 13)}


def parse_unavailable_months(detail: str) -> set[int] | None:
    """item_master의 세부 컬럼("신선과실 / 고정계절품목 - 4월~9월 미출하")에서
    미출하 대상 월(1~12) 집합을 뽑아낸다. 정확한 파싱이 어려우면 None."""
    if not isinstance(detail, str) or "미출하" not in detail:
        return None
    months = set()
    # "4월~9월", "9월~익년2월" 같은 구간, 쉼표로 여러 구간이 있을 수 있음
    for seg in re.findall(r"(\d{1,2})월\s*~\s*(익년)?(\d{1,2})월", detail):
        start_m = int(seg[0])
        end_m = int(seg[2])
        wraps = bool(seg[1])
        if not wraps and start_m <= end_m:
            months.update(range(start_m, end_m + 1))
        else:
            # 연도를 넘어가는 구간 (예: 9월~익년2월)
            months.update(range(start_m, 13))
            months.update(range(1, end_m + 1))
    return months or None


def forecast_tier_b_normal(hist: pd.Series) -> dict:
    """고정계절이 아니거나, 고정계절이지만 출하기간인 경우.

    STL 분해로 추세를 외삽하는 방식은 신선식품처럼 변동성이 큰 시계열에서
    경계(가장 최근 시점) 추세 추정이 불안정해 계절성분과 이중으로 겹쳐
    비현실적인 값을 낼 수 있어(검증 중 확인됨), 대신 "최근 5개년 동안
    이 시기(전월->해당월)에 실제로 몇 % 변했는지의 중앙값"을 쓰는
    직접적인 계절 모멘텀 방식을 기본으로 쓴다. 이 쪽이 이상치에도 강건함."""
    hist = hist.dropna()
    if len(hist) < 25:
        return forecast_tier_a(hist)  # 데이터 부족하면 일반 방식으로 폴백

    last_val = hist.iloc[-1]
    n = len(hist)
    moms = []
    for years_back in range(1, 6):
        idx_prev = -(12 * years_back + 1)
        idx_cur = -(12 * years_back)
        if n + idx_prev < 0:
            break
        prev = hist.iloc[idx_prev]
        cur = hist.iloc[idx_cur]
        if prev and prev != 0 and np.isfinite(prev) and np.isfinite(cur):
            moms.append(cur / prev - 1)

    if len(moms) >= 2:
        seasonal_mom = float(np.median(moms))
    elif moms:
        seasonal_mom = float(moms[0])
    else:
        seasonal_mom = 0.0

    overall_recent = hist.pct_change().dropna()
    overall_recent_mom = float(overall_recent.iloc[-12:].mean()) if len(overall_recent) >= 6 else 0.0

    blended_mom = 0.8 * seasonal_mom + 0.2 * overall_recent_mom
    # 과도한 극단치 방지: 개별 신선식품이라도 한 달에 ±40%를 넘는 경우는 드물다
    blended_mom = float(np.clip(blended_mom, -0.4, 0.4))

    forecast_val = last_val * (1 + blended_mom)
    return {"forecast": forecast_val, "method": "seasonal_median",
            "mom_pct": blended_mom * 100, "n_years_used": len(moms)}


def forecast_tier_b(hist: pd.Series, detail: str, target_ym: str,
                     group_fallback_mom: float | None) -> dict:
    unavailable = parse_unavailable_months(detail)
    target_m = target_month_number(target_ym)
    if unavailable and target_m in unavailable and group_fallback_mom is not None:
        # 통계청 공식 "대체" 로직: 같은 신선분류군의 비계절품목 평균 모멘텀 적용
        last_val = hist.dropna().iloc[-1]
        forecast_val = last_val * (1 + group_fallback_mom)
        return {"forecast": forecast_val, "method": "seasonal_substitute",
                "mom_pct": group_fallback_mom * 100}
    return forecast_tier_b_normal(hist)


# ---------------------------------------------------------------------------
# Tier A 보조: 장기추세만 신뢰할 수 있는 외부지표(K-apt 공동주택관리비 등)를
# "과거 동월 변동치(계절 패턴) + 외부지표 장기추세" 조합으로 반영
# ---------------------------------------------------------------------------


def _seasonal_median_mom(hist: pd.Series, n_years: int = 5) -> float | None:
    """Tier B와 동일한 방식: 최근 n_years개년의 '전월->해당월' 변동률 중앙값."""
    hist = hist.dropna()
    n = len(hist)
    moms = []
    for years_back in range(1, n_years + 1):
        idx_prev = -(12 * years_back + 1)
        idx_cur = -(12 * years_back)
        if n + idx_prev < 0:
            break
        prev, cur = hist.iloc[idx_prev], hist.iloc[idx_cur]
        if prev and prev != 0 and np.isfinite(prev) and np.isfinite(cur):
            moms.append(cur / prev - 1)
    if not moms:
        return None
    return float(np.median(moms))


def _trend_mom_from_external(ext_hist: pd.Series, months_back: int = 12) -> float | None:
    """외부지표의 최근 YoY 성장률을 월간 등가율로 환산(레벨은 믿을만하지만
    MoM 노이즈가 큰 지표를 '현재 장기추세 속도'만 뽑아 쓰는 용도)."""
    ext_hist = ext_hist.dropna()
    if len(ext_hist) < months_back + 1:
        return None
    yoy = ext_hist.iloc[-1] / ext_hist.iloc[-1 - months_back] - 1
    if not np.isfinite(yoy):
        return None
    return (1 + yoy) ** (1 / months_back) - 1


# CPI 품목명 -> 장기추세만 참고하는 외부지표 소스 파일의 컬럼명
# (레벨상관은 높지만 MoM 노이즈가 커서 forecast_seasonal_plus_trend로만 사용)
TREND_ANCHOR_ITEMS = {
    "공동주택관리비": "공용관리비",
}


def forecast_seasonal_plus_trend(hist: pd.Series, ext_hist: pd.Series | None,
                                  seasonal_weight: float = 0.6) -> dict:
    """과거 동월 변동치(계절패턴, CPI 자체) + 외부지표 장기추세를 가중평균해서 예측.
    K-apt처럼 '레벨은 CPI와 거의 같이 움직이지만(상관 0.9+) MoM은 노이즈가 커서
    직접 회귀변수로 못 쓰는' 지표를 그 지표 자체의 MoM이 아니라 스무딩된
    장기추세(YoY -> 월간 등가율)만 뽑아 쓴다."""
    hist = hist.dropna()
    last_val = hist.iloc[-1]
    seasonal_mom = _seasonal_median_mom(hist)
    trend_mom = _trend_mom_from_external(ext_hist) if ext_hist is not None else None

    if seasonal_mom is None and trend_mom is None:
        return forecast_tier_a(hist)
    if trend_mom is None:
        blended = seasonal_mom
        method = "seasonal_only(no_external_trend)"
    elif seasonal_mom is None:
        blended = trend_mom
        method = "external_trend_only"
    else:
        blended = seasonal_weight * seasonal_mom + (1 - seasonal_weight) * trend_mom
        method = f"seasonal({seasonal_weight:.0%})+external_trend({1-seasonal_weight:.0%})"

    forecast_val = last_val * (1 + blended)
    return {"forecast": forecast_val, "method": method, "mom_pct": blended * 100,
            "seasonal_mom_pct": seasonal_mom * 100 if seasonal_mom is not None else None,
            "external_trend_mom_pct": trend_mom * 100 if trend_mom is not None else None}


# ---------------------------------------------------------------------------
# Tier C: 정기개정형 (계단식) - 기본 보합, 수동 오버라이드 지원
# ---------------------------------------------------------------------------


def forecast_tier_c(hist: pd.Series, item_code: str, overrides: dict) -> dict:
    hist = hist.dropna()
    last_val = hist.iloc[-1]
    if item_code in overrides:
        ov = overrides[item_code]
        mom = ov.get("mom_pct", 0.0) / 100
        forecast_val = last_val * (1 + mom)
        return {"forecast": forecast_val, "method": "manual_override",
                "mom_pct": mom * 100, "note": ov.get("note", "")}
    # 최근 개정 이후 경과월 수(참고용 메타데이터)
    changes = hist[hist.diff() != 0]
    months_since_change = len(hist) - 1 - hist.index.get_loc(changes.index[-1]) if len(changes) else None
    return {"forecast": last_val, "method": "carry_forward", "mom_pct": 0.0,
            "months_since_last_change": months_since_change}


# ---------------------------------------------------------------------------
# Tier D: 원자재연동형
# ---------------------------------------------------------------------------

# CPI 품목명 -> 외부데이터(오피넷) 컬럼명 매핑
TIER_D_EXTERNAL_MAP = {
    "휘발유": "보통휘발유_원/L",
    "경유": "자동차용경유_원/L",
    "등유": "실내등유_원/L",
    "자동차용LPG": "자동차용LPG_원/L",
}

# 규제요금 성격이라 시장데이터로 못 잡는 3개 품목 -> Tier C처럼 보합 처리
TIER_D_REGULATED = {"전기료", "도시가스", "지역난방비"}


def forecast_tier_d(item_name: str, hist: pd.Series, external_panel: pd.DataFrame,
                     item_code: str, overrides: dict) -> dict:
    hist = hist.dropna()
    last_val = hist.iloc[-1]

    if item_name in TIER_D_REGULATED:
        return forecast_tier_c(hist, item_code, overrides)

    ext_col = TIER_D_EXTERNAL_MAP.get(item_name)
    if ext_col and ext_col in external_panel.columns:
        ext = external_panel.set_index("date")[ext_col].dropna()
        # 공통 구간에서 beta(외부지표 MoM% -> CPI품목 MoM%) 회귀
        cpi_mom = mom_series(hist) / 100
        ext_mom = mom_series(ext) / 100
        common = cpi_mom.index.intersection(ext_mom.index)
        common = [c for c in common if pd.notna(cpi_mom.get(c)) and pd.notna(ext_mom.get(c))]
        if len(common) >= 12:
            y = cpi_mom.loc[common].values
            x = ext_mom.loc[common].values
            beta = np.sum(x * y) / np.sum(x * x) if np.sum(x * x) != 0 else 1.0
            # 외부지표 자체의 다음달 값은 최근 3개월 평균 모멘텀으로 단순 외삽
            ext_recent_mom = ext_mom.iloc[-3:].mean()
            pred_cpi_mom = beta * ext_recent_mom
            forecast_val = last_val * (1 + pred_cpi_mom)
            return {"forecast": forecast_val, "method": f"external_regression({ext_col})",
                    "mom_pct": pred_cpi_mom * 100, "beta": beta}

    # 외부데이터 없는 취사용LPG/부탄가스 등은 일반 시장가격형 방식으로 폴백
    return forecast_tier_a(hist)
