"""
대분류(12) 대신 소분류(38, "01.1 식료품" 등) 단위로 탑다운 vs 바텀업을 24개월
워크포워드 비교. 대분류 하이브리드와 같은 방법론(backtest_topdown_vs_bottomup_category.py)을
한 단계 더 세분화한 버전 - 사용자 제안("지출목적 안에 소분류로 나눠서") 검증.

코드 매핑: CPI_Tier_분류.csv의 품목코드 앞 3글자(예: A011010 -> A01)가 소분류코드다.
대분류 letter 순서(A~L)와 소분류코드 정렬순서가 공식 라벨("01.1","01.2"...) 순서와
정확히 1:1 대응함을 실측으로 확인했다(단 12.2에 해당하는 L03코드만 공식 라벨이
없어 매핑 제외 - 해당 품목은 바텀업만 사용).
"""
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

SCRIPTS = Path(__file__).resolve().parent
LETTER_ORDER = list("ABCDEFGHIJKL")


def build_subcat_map():
    tier = pd.read_csv(SCRIPTS.parent / "CPI_Tier_분류.csv", encoding="utf-8-sig")
    tier.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"]
    tier["소분류코드"] = tier["품목코드"].str[:3]
    tier["대분류letter"] = tier["품목코드"].str[0]

    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = official.columns[0]
    sub_labels_all = official[official[item_col].str.match(r"^\d\d\.\d ")][item_col].tolist()

    code_to_label = {}
    for letter in LETTER_ORDER:
        codes = sorted(tier[tier["대분류letter"] == letter]["소분류코드"].unique())
        major_num = LETTER_ORDER.index(letter) + 1
        labels = sorted([l for l in sub_labels_all if l.startswith(f"{major_num:02d}.")],
                         key=lambda x: int(x.split(".")[1].split(" ")[0]))
        if len(codes) != len(labels):
            print(f"[경고] {letter}({major_num}): 코드 {len(codes)}개 vs 라벨 {len(labels)}개 - 불일치, 매핑 가능한 만큼만 순서대로 연결")
        for c, l in zip(codes, labels):
            code_to_label[c] = l

    tier["소분류"] = tier["소분류코드"].map(code_to_label)
    return tier[["품목코드", "품목명", "가중치", "소분류코드", "소분류"]]


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
    subcat_map = build_subcat_map()
    print(f"소분류 매핑: {subcat_map['소분류'].notna().sum()}/{len(subcat_map)}개 품목 매핑됨 "
          f"(미매핑 {subcat_map['소분류'].isna().sum()}개는 소분류 없음/바텀업 전용)")

    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = official.columns[0]
    official = official.drop_duplicates(subset=item_col, keep="first")
    month_cols_sorted = sorted([c for c in official.columns if c[:2] in ("19", "20")])
    sub_rows = official[official[item_col].str.match(r"^\d\d\.\d ")]

    bu = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv", encoding="utf-8-sig")
    bu_cols = bu.columns.tolist()
    name_col, weight_col = bu_cols[1], bu_cols[3]
    bu = bu.merge(subcat_map[["품목명", "소분류", "가중치"]].rename(columns={"가중치": "가중치2"}),
                  left_on=name_col, right_on="품목명", how="left")
    bu["소분류"] = bu["소분류"]

    test_months = sorted([c.replace("pred_", "") for c in bu_cols if c.startswith("pred_")])
    print(f"백테스트 대상 월: {test_months[0]} ~ {test_months[-1]} ({len(test_months)}개월)")

    records = []
    for m in test_months:
        train_cols = [c for c in month_cols_sorted if c < m]
        for _, r in sub_rows.iterrows():
            sub = r[item_col]
            series = pd.Series({c: r[c] for c in month_cols_sorted}).astype(float)
            hist = series[train_cols].dropna()
            actual = series.get(m, np.nan)
            if pd.isna(actual) or len(hist) < 36:
                continue
            pred_td = ets_forecast(hist, is_seasonal=True)
            records.append({"월": m, "소분류": sub, "방식": "탑다운", "예측": pred_td, "실제": actual})

        pred_col, actual_col = f"pred_{m}", f"actual_{m}"
        subm = bu[[weight_col, "소분류", pred_col]].dropna()
        for sub, g in subm.groupby("소분류"):
            pred_bu = blend(g[pred_col].values, g[weight_col].values)
            actual_row = sub_rows.loc[sub_rows[item_col] == sub, m]
            if len(actual_row) == 0 or pd.isna(actual_row.iloc[0]):
                continue
            records.append({"월": m, "소분류": sub, "방식": "바텀업", "예측": pred_bu, "실제": actual_row.iloc[0]})

    out = pd.DataFrame(records)
    out["오차"] = out["예측"] - out["실제"]
    out["오차%"] = out["오차"] / out["실제"] * 100
    out.to_csv(SCRIPTS / "topdown_vs_bottomup_subcat_backtest.csv", index=False, encoding="utf-8-sig")

    summary = out.groupby(["소분류", "방식"]).agg(
        MAE=("오차", lambda x: x.abs().mean()),
        MAPE=("오차%", lambda x: x.abs().mean()),
        n=("오차", "count"),
    ).reset_index()
    summary.to_csv(SCRIPTS / "topdown_vs_bottomup_subcat_summary.csv", index=False, encoding="utf-8-sig")
    return out, summary


if __name__ == "__main__":
    run()
