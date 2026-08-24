"""
2026년 8월 CPI 예측 - 두 갈래 교차검증.

A) 바텀업: 기존 458개 품목 파이프라인(forecast_all_tiers.py 로직 재사용)을 2026-08
   1개월에 대해 실행. Tier D 5개 품목은 8월 오피넷 실측(부분월) 가격을 회귀변수로 사용.
B) 탑다운: 지출목적별 대분류(12)·소분류(38) 각각을 그 자체의 2000-01~2026-07 히스토리로
   직접 ETS 적합해 8월을 예측(개별 leaf item과 동일한 기법, Tier A에서 검증된 방식).

두 결과가 서로 비슷하게 나오는지로 교차검증한다.
"""
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"
TARGET = "2026-08"

FIXED_SEASON_NAMES = {
    "복숭아", "포도", "감", "귤", "오렌지", "참외", "수박", "딸기", "체리", "열무", "굴",
}
ANNUAL_STEP_NAMES = {
    "외래진료비", "한방진료비", "약국조제료", "치과진료비", "입원진료비",
    "전문대학납입금", "국공립대학교납입금", "사립대학교납입금",
    "국공립대학원납입금", "사립대학원납입금",
}
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
        extracted = df["기간"].str.extract(r"(\d{4})년(\d{2})월")
        df["기간"] = extracted[0] + "-" + extracted[1]
    else:
        # "2026-08(부분월,...)" 같은 라벨을 "2026-08"로 정규화
        df["기간"] = df["기간"].str.extract(r"(\d{4}-\d{2})")[0]
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


def regression_forecast(cpi_hist: pd.Series, ext_hist: pd.Series, ext_target_value: float) -> float:
    common = sorted(set(cpi_hist.dropna().index) & set(ext_hist.dropna().index))
    if len(common) < 24 or pd.isna(ext_target_value):
        drift = cpi_hist.dropna().diff().dropna().tail(12).mean()
        return float(cpi_hist.dropna().iloc[-1] + (drift if pd.notna(drift) else 0))
    y = cpi_hist[common].astype(float).values
    x = ext_hist[common].astype(float).values
    b, a = np.polyfit(x, y, 1)
    return float(a + b * ext_target_value)


def bottom_up():
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv")
    month_cols_sorted = sorted([c for c in panel.columns if c[:2] in ("19", "20")])
    train_cols = [c for c in month_cols_sorted if c < TARGET]

    opinet_cache = {name: load_opinet_series(f, c) for name, (f, c) in OPINET_REGRESSOR.items()}

    rows = []
    for _, row in panel.iterrows():
        item, tier, weight = row["품목명"], row["Tier"], row["가중치"]
        full_series = pd.Series(row[month_cols_sorted].astype(float).values,
                                 index=month_cols_sorted).interpolate(limit_area="inside")
        cpi_hist = full_series[train_cols]

        if item in OPINET_REGRESSOR:
            ext_series = opinet_cache[item]
            ext_hist = ext_series[ext_series.index.isin(train_cols)]
            ext_target = ext_series.get(TARGET, np.nan)
            pred = regression_forecast(cpi_hist, ext_hist, ext_target)
            model_used = "회귀(오피넷 8월실측)"
        else:
            if tier == "B":
                is_seasonal, restrict = True, item in FIXED_SEASON_NAMES
            elif tier == "C" and item in ANNUAL_STEP_NAMES:
                is_seasonal, restrict = True, False
            else:
                is_seasonal, restrict = False, False
            hist = cpi_hist[cpi_hist.index >= "2017-01"] if restrict else cpi_hist
            pred = ets_forecast(hist, is_seasonal)
            model_used = "계절ETS" if is_seasonal else "비계절ETS"

        rows.append({"품목코드": row["품목코드"], "품목명": item, "Tier": tier,
                      "가중치": weight, "모델": model_used, "pred_2026-08": pred})

    out = pd.DataFrame(rows)
    out.to_csv(SCRIPTS / "august2026_bottomup_items.csv", index=False, encoding="utf-8-sig")

    w = out["가중치"].values
    vals = out["pred_2026-08"].astype(float).values
    arith = np.average(vals, weights=w)
    geom = np.exp(np.average(np.log(vals), weights=w))
    blended = 0.5 * arith + 0.5 * geom
    return out, blended


def top_down():
    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv").drop_duplicates(subset="품목", keep="first")
    month_cols_sorted = sorted([c for c in official.columns if c[:2] in ("19", "20")])
    train_cols = [c for c in month_cols_sorted if c < TARGET]

    total_row = official[official["품목"] == "0 총지수"].iloc[0]
    major_rows = official[official["품목"].str.match(r"^\d{2} ")]
    sub_rows = official[official["품목"].str.match(r"^\d{2}\.\d ")]

    results = []
    for label, r in [("총지수", total_row)] + list(zip(major_rows["품목"], major_rows.to_dict("records"))) + \
                     list(zip(sub_rows["품목"], sub_rows.to_dict("records"))):
        series = pd.Series({c: r[c] for c in month_cols_sorted}).astype(float)
        hist = series[train_cols].dropna()
        # 대분류/소분류는 leaf item을 집계한 것이라 계절성이 뚜렷한 경우가 많음
        # (예: 11.2 숙박서비스는 26년 내내 7월->8월에 항상 상승) -> 계절ETS 기본 적용
        pred = ets_forecast(hist, is_seasonal=True)
        last_actual = hist.iloc[-1] if len(hist) else np.nan
        results.append({
            "분류": label, "2026-07(실제)": last_actual, "2026-08(예측)": pred,
            "전월대비%": (pred - last_actual) / last_actual * 100 if last_actual else np.nan,
        })

    out = pd.DataFrame(results)
    out.to_csv(SCRIPTS / "august2026_topdown_categories.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    print("=== A) 바텀업(458개 품목 -> 블렌드 가중집계) ===")
    items_df, blended_total = bottom_up()
    print(f"2026-08 예측 총지수(바텀업): {blended_total:.3f}")

    print("\n=== B) 탑다운(대분류12 + 소분류38, 자체 히스토리 ETS) ===")
    cat_df = top_down()
    print(cat_df.to_string(index=False))

    print("\n두 방식 총지수 비교:")
    total_topdown = cat_df[cat_df["분류"] == "총지수"]["2026-08(예측)"].iloc[0]
    print(f"  바텀업: {blended_total:.3f}  /  탑다운(총지수 직접ETS): {total_topdown:.3f}")
