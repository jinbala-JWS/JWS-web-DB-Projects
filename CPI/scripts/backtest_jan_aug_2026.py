"""
현재 확정된 모델(바텀업 458개+게이트 편향보정+아이템별 자동선택 + 탑다운 + 하이브리드
결합)을 2026-01~2026-08 8개월에 대해 워크포워드로 재현하고 실제 발표치와 비교한다.

워크포워드 원칙: 각 대상월 예측 시 그 월 이전 데이터만 사용(전 파이프라인 공통).
단, "이 품목엔 어떤 방식(계절ETS/계단평균/완전동결)을 쓸지"를 정한
item_model_selection.csv / category_method_selection.csv 자체는 2024-08~2026-07(대상월
일부와 겹치는) 24개월 백테스트로 한 번 확정한 고정 설정이다 - 매 대상월마다
"그 방식 선택 자체"를 다시 학습하지는 않는다(하이퍼파라미터 고정 후 워크포워드 평가와
동일한 성격 - 완전한 실시간 재현은 아니라는 점을 결과에서 밝힌다).

바텀업/하이브리드 두 방식 모두 계산해 비교한다.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import forecast_september_2026 as fs
from hybrid_category_model import bottomup_to_category, combine_hybrid, hybrid_total, category_weights

warnings.filterwarnings("ignore")

SCRIPTS = Path(__file__).resolve().parent
TARGET_MONTHS = [f"2026-{m:02d}" for m in range(1, 9)]


def bottom_up_for_month(panel, month_cols_sorted, opinet_cache, item_selection, target):
    train_cols = [c for c in month_cols_sorted if c < target]
    target_month_num = int(target[5:7])
    rows = []
    for _, row in panel.iterrows():
        item, tier, weight = row["품목명"], row["Tier"], row["가중치"]
        full_series = pd.Series(row[month_cols_sorted].astype(float).values,
                                 index=month_cols_sorted).interpolate(limit_area="inside")
        cpi_hist = full_series[train_cols]

        if item in fs.OPINET_REGRESSOR:
            fname, col = fs.OPINET_REGRESSOR[item]
            ext_series = opinet_cache[item]
            ext_hist = ext_series[ext_series.index.isin(train_cols)]
            ext_target = ext_series.get(target, np.nan)
            pred = fs.regression_forecast(cpi_hist, ext_hist, ext_target)
        elif item_selection.get(item) == "계단평균":
            pred = fs.step_weighted_forecast(cpi_hist, target_month_num)
        elif item_selection.get(item) == "완전동결":
            pred = float(cpi_hist.dropna().iloc[-1])
        else:
            if tier == "B":
                is_seasonal, restrict = True, item in fs.FIXED_SEASON_NAMES
            elif tier == "C" and item in fs.ANNUAL_STEP_NAMES:
                is_seasonal, restrict = True, False
            elif item in fs.ANNUAL_SEASONAL_TIER_D:
                is_seasonal, restrict = True, False
            elif item in fs.ANNUAL_SEASONAL_MISC:
                is_seasonal, restrict = True, False
            elif item_selection.get(item) == "계절ETS":
                is_seasonal, restrict = True, False
            else:
                is_seasonal, restrict = False, False
            hist = cpi_hist[cpi_hist.index >= "2017-01"] if restrict else cpi_hist
            pred = fs.ets_forecast(hist, is_seasonal)

        rows.append({"품목코드": row["품목코드"], "품목명": item, "Tier": tier,
                      "가중치": weight, "pred": pred})
    return pd.DataFrame(rows)


def apply_bias_correction_dynamic(bu_items: pd.DataFrame, target: str, backtest: pd.DataFrame) -> pd.DataFrame:
    all_months = sorted({c.replace("pred_", "") for c in backtest.columns if c.startswith("pred_")})
    trailing = [m for m in all_months if m < target][-6:]
    if len(trailing) < 6:
        bu_items["pred_corr"] = bu_items["pred"]
        return bu_items

    corrections = {}
    for _, row in backtest.iterrows():
        item = row["품목명"]
        lvl = [row[f"pred_{m}"] - row[f"actual_{m}"] for m in trailing]
        pct = [(row[f"pred_{m}"] - row[f"actual_{m}"]) / row[f"actual_{m}"] * 100 for m in trailing]
        mean_pct = np.mean(pct)
        consistency = np.mean([np.sign(p) == np.sign(mean_pct) for p in pct])
        std_pct = np.std(pct)
        if consistency >= 0.8 and std_pct <= 2.0 and abs(mean_pct) >= 0.05:
            corrections[item] = np.mean(lvl)
        else:
            corrections[item] = 0.0

    bu_items["보정치"] = bu_items["품목명"].map(corrections).fillna(0.0)
    bu_items["pred_corr"] = bu_items["pred"] - 0.6 * bu_items["보정치"]
    return bu_items


def top_down_for_month(official, item_col, month_cols_sorted, target):
    train_cols = [c for c in month_cols_sorted if c < target]
    total_row = official[official[item_col] == "0 총지수"].iloc[0]
    major_rows = official[official[item_col].str.match(r"^\d{2} ")]

    results = []
    for label, r in [("총지수", total_row)] + list(zip(major_rows[item_col], major_rows.to_dict("records"))):
        series = pd.Series({c: r[c] for c in month_cols_sorted}).astype(float)
        hist = series[train_cols].dropna()
        pred = fs.ets_forecast(hist, is_seasonal=True)
        results.append({"분류": label, "탑다운예측": pred})
    return pd.DataFrame(results)


def blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def main():
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv", encoding="utf-8-sig")
    panel.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"] + list(panel.columns[6:])
    month_cols_sorted = sorted([c for c in panel.columns if c[:2] in ("19", "20")])
    opinet_cache = {name: fs.load_opinet_series(f, c) for name, (f, c) in fs.OPINET_REGRESSOR.items()}
    item_selection = fs.load_item_model_selection()
    backtest = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv")

    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = official.columns[0]
    official = official.drop_duplicates(subset=item_col, keep="first")

    cat_sel = pd.read_csv(SCRIPTS / "category_method_selection.csv", encoding="utf-8-sig")
    cat_sel_map = dict(zip(cat_sel["분류"], cat_sel["채택방식"]))
    w_cat = category_weights()

    ACTUAL_TOTAL = {m: official.loc[official[item_col] == "0 총지수", m].iloc[0] for m in TARGET_MONTHS}

    summary_rows = []
    for target in TARGET_MONTHS:
        print(f"=== {target} 처리중 ===")
        bu_items = bottom_up_for_month(panel, month_cols_sorted, opinet_cache, item_selection, target)
        bu_items = apply_bias_correction_dynamic(bu_items, target, backtest)
        bu_items.to_csv(SCRIPTS / f"backtest_{target}_bottomup_items.csv", index=False, encoding="utf-8-sig")

        bu_total = blend(bu_items["pred_corr"].values, bu_items["가중치"].values)

        td = top_down_for_month(official, item_col, month_cols_sorted, target)
        td_total_row = td[td["분류"] == "총지수"]["탑다운예측"].iloc[0]
        td_major = td[td["분류"].str.match(r"^\d\d ")]

        bu_cat = bottomup_to_category(bu_items, pred_col="pred_corr")
        hybrid = combine_hybrid(bu_cat, td_major.rename(columns={"탑다운예측": "탑다운예측"}), selection=cat_sel_map)
        hyb_total = hybrid_total(hybrid)

        actual = ACTUAL_TOTAL[target]
        summary_rows.append({
            "월": target, "실제총지수": actual,
            "바텀업예측": bu_total, "바텀업오차%": (bu_total - actual) / actual * 100,
            "탑다운예측(총지수자체)": td_total_row, "탑다운오차%": (td_total_row - actual) / actual * 100,
            "하이브리드예측": hyb_total, "하이브리드오차%": (hyb_total - actual) / actual * 100,
        })

    out = pd.DataFrame(summary_rows)
    out.to_csv(SCRIPTS / "backtest_jan_aug_2026_summary.csv", index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))
    print(f"\n바텀업 MAPE(8개월): {out['바텀업오차%'].abs().mean():.4f}%")
    print(f"탑다운 MAPE(8개월): {out['탑다운오차%'].abs().mean():.4f}%")
    print(f"하이브리드 MAPE(8개월): {out['하이브리드오차%'].abs().mean():.4f}%")
    return out


if __name__ == "__main__":
    main()
