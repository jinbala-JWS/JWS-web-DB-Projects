"""
전체 458개 품목(Tier A+B+C+D) — 과거 1년(2025-08~2026-07) 워크포워드 1개월 예측.
매 스텝, 그 시점까지의 "실제" 데이터만 사용해 모델을 재적합(refit)한 뒤 다음 1개월을 예측한다
(미래정보 누출 없음).

품목별 모델 선택:
- Tier A: ETS(완만감쇠 추세, 비계절)
- Tier B: ETS(완만감쇠 추세 + 월별 계절성). 고정계절 11개 품목은 2017-01 이후만 사용
- Tier C:
  - 연 1회 정해진 달에 계단식으로 바뀌는 10개 품목(진료비 5종=매년1월, 등록금 5종=매년3월)은
    계절ETS 적용(연 1회 패턴도 "계절성"으로 근사)
  - 나머지 21개 품목은 비계절 ETS
- Tier D:
  - 오피넷 실측 데이터와 상관계수가 높게 검증된 5개 품목(휘발유·경유·등유·자동차용LPG·취사용LPG)은
    "해당월의 실제 오피넷 평균가"를 회귀변수로 쓰는 선형회귀 모델 사용
    (Tier D 설계원칙: 외부데이터가 있으면 우선 사용 - 오피넷 가격은 월중 집계되어 CPI 발표
    전에 이미 알 수 있는 선행정보이므로 회귀변수로 쓰는 것이 정당함)
  - 나머지(부탄가스·전기료·도시가스·지역난방비)는 비계절 ETS (연속 외부데이터 없음)

데이터 부족(24개월 미만)/적합 실패 시 최근 12개월 평균변화율 기반 랜덤워크로 대체.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"

TARGET_MONTHS = [
    "2025-08", "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
]

FIXED_SEASON_NAMES = {
    "복숭아", "포도", "감", "귤", "오렌지", "참외", "수박", "딸기", "체리", "열무", "굴",
}
ANNUAL_STEP_NAMES = {
    "외래진료비", "한방진료비", "약국조제료", "치과진료비", "입원진료비",
    "전문대학납입금", "국공립대학교납입금", "사립대학교납입금",
    "국공립대학원납입금", "사립대학원납입금",
}
# Tier D: 오피넷 실측치를 회귀변수로 사용하는 5개 품목 -> (오피넷 TSV파일, 오피넷 컬럼명)
OPINET_REGRESSOR = {
    "휘발유": ("raw_opinet_gasoline_diesel_kerosene.tsv", "보통휘발유"),
    "경유": ("raw_opinet_gasoline_diesel_kerosene.tsv", "자동차용경유"),
    "등유": ("raw_opinet_gasoline_diesel_kerosene.tsv", "실내등유"),
    "자동차용LPG": ("raw_opinet_auto_lpg.tsv", "자동차부탄(원/L)"),
    "취사용LPG": ("raw_opinet_household_lpg.tsv", "일반프로판(원/kg)"),
}


def load_opinet_series(fname, col):
    df = pd.read_csv(SCRIPTS / fname, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "기간"})
    if "년" in str(df["기간"].iloc[0]):
        df["기간"] = df["기간"].str.extract(r"(\d{4})년(\d{2})월")[0] + "-" + df["기간"].str.extract(r"(\d{4})년(\d{2})월")[1]
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    return df.set_index("기간")[col]


def ets_forecast(hist: pd.Series, is_seasonal: bool) -> float:
    s = hist.dropna()
    if len(s) < 24:
        if len(s) >= 2:
            drift = s.diff().dropna().tail(12).mean()
            return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))
        return float(s.iloc[-1]) if len(s) else np.nan
    try:
        if is_seasonal and len(s) >= 36:
            model = ExponentialSmoothing(
                s.values, trend="add", damped_trend=True,
                seasonal="add", seasonal_periods=12, initialization_method="estimated",
            )
        else:
            model = ExponentialSmoothing(
                s.values, trend="add", damped_trend=True, seasonal=None,
                initialization_method="estimated",
            )
        fit = model.fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        drift = s.diff().dropna().tail(12).mean()
        return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))


def regression_forecast(cpi_hist: pd.Series, ext_hist: pd.Series, ext_target_value: float) -> float:
    """CPI_index ~ a + b*외부가격 선형회귀를 과거데이터로 적합해 목표월 외부가격으로 예측."""
    common = sorted(set(cpi_hist.dropna().index) & set(ext_hist.dropna().index))
    if len(common) < 24 or pd.isna(ext_target_value):
        drift = cpi_hist.dropna().diff().dropna().tail(12).mean()
        return float(cpi_hist.dropna().iloc[-1] + (drift if pd.notna(drift) else 0))
    y = cpi_hist[common].astype(float).values
    x = ext_hist[common].astype(float).values
    b, a = np.polyfit(x, y, 1)
    return float(a + b * ext_target_value)


def main():
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv")
    month_cols_sorted = sorted([c for c in panel.columns if c[:2] in ("19", "20")])

    opinet_cache = {name: load_opinet_series(f, c) for name, (f, c) in OPINET_REGRESSOR.items()}

    results = []
    n = len(panel)
    for i, row in panel.iterrows():
        item = row["품목명"]
        tier = row["Tier"]
        weight = row["가중치"]

        full_series = pd.Series(
            row[month_cols_sorted].astype(float).values, index=month_cols_sorted
        ).interpolate(limit_area="inside")

        use_regression = item in OPINET_REGRESSOR
        if tier == "B":
            is_seasonal = True
            restrict_2017 = item in FIXED_SEASON_NAMES
        elif tier == "C" and item in ANNUAL_STEP_NAMES:
            is_seasonal = True
            restrict_2017 = False
        else:
            is_seasonal = False
            restrict_2017 = False

        preds = {}
        for target in TARGET_MONTHS:
            train_end_idx = month_cols_sorted.index(target) - 1
            train_cols = month_cols_sorted[: train_end_idx + 1]
            cpi_hist = full_series[train_cols]

            if use_regression:
                ext_series = opinet_cache[item]
                ext_hist = ext_series[ext_series.index.isin(train_cols)]
                ext_target = ext_series.get(target, np.nan)
                pred = regression_forecast(cpi_hist, ext_hist, ext_target)
            else:
                hist = cpi_hist
                if restrict_2017:
                    hist = hist[hist.index >= "2017-01"]
                pred = ets_forecast(hist, is_seasonal=is_seasonal)
            preds[target] = pred

        rec = {"품목코드": row["품목코드"], "품목명": item, "Tier": tier, "가중치": weight,
               "모델": "회귀(오피넷)" if use_regression else ("계절ETS" if is_seasonal else "비계절ETS")}
        for target in TARGET_MONTHS:
            rec[f"pred_{target}"] = preds[target]
            rec[f"actual_{target}"] = full_series.get(target, np.nan)
        results.append(rec)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} 품목 처리 완료 ({item})")

    out = pd.DataFrame(results)
    out.to_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv", index=False, encoding="utf-8-sig")
    print("저장:", SCRIPTS / "all_tiers_forecast_vs_actual.csv", out.shape)


if __name__ == "__main__":
    main()
