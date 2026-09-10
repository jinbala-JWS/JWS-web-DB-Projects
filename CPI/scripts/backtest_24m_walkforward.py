"""
현재 확정된 모델(바텀업+게이트편향보정+품목별자동선택 / 하이브리드)을 최근 24개월
(2024-09~2026-08, 실제 발표치가 확보된 가장 최근 24개월)에 워크포워드로 재현하고
실제와 비교. backtest_jan_aug_2026.py의 재사용 가능한 부품을 그대로 쓰고 대상
기간만 24개월로 확장한 버전.
"""
import pandas as pd
from pathlib import Path

from backtest_jan_aug_2026 import (
    bottom_up_for_month, apply_bias_correction_dynamic, top_down_for_month, blend,
)
import forecast_september_2026 as fs
from hybrid_category_model import bottomup_to_category, combine_hybrid, hybrid_total, category_weights

SCRIPTS = Path(__file__).resolve().parent
TARGET_MONTHS = [f"2024-{m:02d}" for m in range(9, 13)] + [f"2025-{m:02d}" for m in range(1, 13)] + \
                [f"2026-{m:02d}" for m in range(1, 9)]


def main():
    print(f"대상: {TARGET_MONTHS[0]} ~ {TARGET_MONTHS[-1]} ({len(TARGET_MONTHS)}개월)")

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

    ACTUAL_TOTAL = {m: official.loc[official[item_col] == "0 총지수", m].iloc[0] for m in TARGET_MONTHS}

    summary_rows = []
    for target in TARGET_MONTHS:
        print(f"=== {target} 처리중 ===", flush=True)
        bu_items = bottom_up_for_month(panel, month_cols_sorted, opinet_cache, item_selection, target)
        bu_items = apply_bias_correction_dynamic(bu_items, target, backtest)
        bu_items.to_csv(SCRIPTS / f"backtest24m_{target}_bottomup_items.csv", index=False, encoding="utf-8-sig")

        bu_total = blend(bu_items["pred_corr"].values, bu_items["가중치"].values)

        td = top_down_for_month(official, item_col, month_cols_sorted, target)
        td_total_row = td[td["분류"] == "총지수"]["탑다운예측"].iloc[0]
        td_major = td[td["분류"].str.match(r"^\d\d ")]

        bu_cat = bottomup_to_category(bu_items, pred_col="pred_corr")
        hybrid = combine_hybrid(bu_cat, td_major, selection=cat_sel_map)
        hyb_total = hybrid_total(hybrid)

        actual = ACTUAL_TOTAL[target]
        summary_rows.append({
            "월": target, "실제총지수": actual,
            "바텀업예측": bu_total, "바텀업오차%": (bu_total - actual) / actual * 100,
            "탑다운예측(총지수자체)": td_total_row, "탑다운오차%": (td_total_row - actual) / actual * 100,
            "하이브리드예측": hyb_total, "하이브리드오차%": (hyb_total - actual) / actual * 100,
        })

    out = pd.DataFrame(summary_rows)
    out.to_csv(SCRIPTS / "backtest_24m_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    print(out.round(4).to_string(index=False))
    print(f"\n바텀업 MAPE(24개월): {out['바텀업오차%'].abs().mean():.4f}%")
    print(f"탑다운 MAPE(24개월): {out['탑다운오차%'].abs().mean():.4f}%")
    print(f"하이브리드 MAPE(24개월): {out['하이브리드오차%'].abs().mean():.4f}%")
    print("DONE")
    return out


if __name__ == "__main__":
    main()
