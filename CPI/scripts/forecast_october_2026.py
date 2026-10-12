"""
2026년 10월 CPI 예측(상반월 시점) — 월중(10/1~10/10 오피넷 MTD 반영) 예측.
백테스트에서 쓰는 월 파라미터화 부품(backtest_jan_aug_2026.py)을 그대로 재사용:
바텀업 458개(품목별 자동선택·이상치보정) -> 게이트 편향보정(직전 6개월 동적) -> 탑다운 ->
하이브리드 결합. 학습데이터는 2026-09 실적(KOSIS OpenAPI)까지 포함.

사용 전 준비: python fetch_kosis_cpi.py 202609, python append_opinet_mtd.py 2026-10
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import forecast_september_2026 as fs
from backtest_jan_aug_2026 import (
    bottom_up_for_month, apply_bias_correction_dynamic, top_down_for_month, blend,
)
from hybrid_category_model import bottomup_to_category, combine_hybrid, hybrid_total

warnings.filterwarnings("ignore")
SCRIPTS = Path(__file__).resolve().parent
TARGET = "2026-10"
PREV = "2026-09"
YEAR_AGO = "2025-10"


def main():
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv", encoding="utf-8-sig")
    panel.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"] + list(panel.columns[6:])
    month_cols = sorted([c for c in panel.columns if c[:2] in ("19", "20")])
    assert PREV in month_cols, f"{PREV} 실적이 패널에 없음 - fetch_kosis_cpi.py 먼저 실행"
    opinet_cache = {n: fs.load_opinet_series(f, c) for n, (f, c) in fs.OPINET_REGRESSOR.items()}
    item_selection = fs.load_item_model_selection()
    backtest = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv")

    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = official.columns[0]
    official = official.drop_duplicates(subset=item_col, keep="first")

    bu = bottom_up_for_month(panel, month_cols, opinet_cache, item_selection, TARGET)
    bu = apply_bias_correction_dynamic(bu, TARGET, backtest)
    bu.to_csv(SCRIPTS / "october2026_bottomup_items.csv", index=False, encoding="utf-8-sig")
    n_corr = int((bu.get("보정치", 0) != 0).sum()) if "보정치" in bu else 0
    print(f"편향보정 적용 품목: {n_corr}/{len(bu)}")

    bu_total = blend(bu["pred_corr"].values, bu["가중치"].values)
    td = top_down_for_month(official, item_col, month_cols, TARGET)
    td_total = td.loc[td["분류"] == "총지수", "탑다운예측"].iloc[0]
    td_major = td[td["분류"].str.match(r"^\d\d ")]

    cat_sel = pd.read_csv(SCRIPTS / "category_method_selection.csv", encoding="utf-8-sig")
    hybrid = combine_hybrid(bottomup_to_category(bu, pred_col="pred_corr"), td_major,
                            selection=dict(zip(cat_sel["분류"], cat_sel["채택방식"]))).sort_values("분류")
    hyb_total = hybrid_total(hybrid)

    prev = {r[item_col]: r[PREV] for _, r in official.iterrows()}
    ago = {r[item_col]: r[YEAR_AGO] for _, r in official.iterrows()}
    hybrid["9월실제"] = hybrid["분류"].map(prev)
    hybrid["작년10월"] = hybrid["분류"].map(ago)
    hybrid["전월비%"] = (hybrid["최종예측"] / hybrid["9월실제"] - 1) * 100
    hybrid["전년비%"] = (hybrid["최종예측"] / hybrid["작년10월"] - 1) * 100
    hybrid.to_csv(SCRIPTS / "october2026_hybrid_category.csv", index=False, encoding="utf-8-sig")

    p, a = prev["0 총지수"], ago["0 총지수"]
    rows = [("바텀업", bu_total), ("탑다운", td_total), ("하이브리드(공식)", hyb_total)]
    out = pd.DataFrame([{"방식": n, "10월예측": v, "9월실제": p, "전월비%": (v / p - 1) * 100,
                          "작년10월": a, "전년비%": (v / a - 1) * 100} for n, v in rows])
    out.to_csv(SCRIPTS / "october2026_total_summary.csv", index=False, encoding="utf-8-sig")
    print(out.round(3).to_string(index=False))
    print(hybrid[["분류", "채택방식", "최종예측", "전월비%", "전년비%"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
