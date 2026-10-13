"""
2026년 9월 CPI 실제 발표치(KOSIS, 2026-10-02 확인) vs 우리 예측치(하이브리드/바텀업/
탑다운) 오차 검증 - evaluate_august_actual.py의 9월판.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

ACTUAL_SEP = {
    "0 총지수": 120.43,
    "01 식료품 및 비주류음료": 130.09,
    "02 주류 및 담배": 105.19,
    "03 의류 및 신발": 119.74,
    "04 주택, 수도, 전기 및 연료": 118.93,
    "05 가정용품 및 가사 서비스": 122.26,
    "06 보건": 107.22,
    "07 교통": 124.13,
    "08 통신": 103.42,
    "09 오락 및 문화": 115.93,
    "10 교육": 110.34,
    "11 음식 및 숙박": 128.52,
    "12 기타 상품 및 서비스": 130.36,
}
ACTUAL_AUG = {
    "0 총지수": 120.05,
    "01 식료품 및 비주류음료": 128.27, "02 주류 및 담배": 105.29, "03 의류 및 신발": 119.72,
    "04 주택, 수도, 전기 및 연료": 117.59, "05 가정용품 및 가사 서비스": 123.16, "06 보건": 107.04,
    "07 교통": 124.63, "08 통신": 102.48, "09 오락 및 문화": 115.80, "10 교육": 110.31,
    "11 음식 및 숙박": 129.03, "12 기타 상품 및 서비스": 130.45,
}


def main():
    hybrid = pd.read_csv(SCRIPTS / "september2026_hybrid_category.csv", encoding="utf-8-sig")
    hybrid["실제(9월)"] = hybrid["분류"].map(ACTUAL_SEP)
    hybrid["실제(8월)"] = hybrid["분류"].map(ACTUAL_AUG)
    hybrid["실제_전월대비%"] = (hybrid["실제(9월)"] - hybrid["실제(8월)"]) / hybrid["실제(8월)"] * 100

    hybrid["최종오차"] = hybrid["최종예측"] - hybrid["실제(9월)"]
    hybrid["최종오차%"] = hybrid["최종오차"] / hybrid["실제(9월)"] * 100
    hybrid["바텀업오차%"] = (hybrid["바텀업예측"] - hybrid["실제(9월)"]) / hybrid["실제(9월)"] * 100
    hybrid["탑다운오차%"] = (hybrid["탑다운예측"] - hybrid["실제(9월)"]) / hybrid["실제(9월)"] * 100

    # 총지수 추가
    bu_items = pd.read_csv(SCRIPTS / "september2026_bottomup_items_corrected.csv", encoding="utf-8-sig")
    w = bu_items["가중치"].values
    vals = bu_items["보정후_pred_2026-09"].values
    arith = np.average(vals, weights=w)
    geom = np.exp(np.average(np.log(vals), weights=w))
    bu_total = 0.5 * arith + 0.5 * geom

    td = pd.read_csv(SCRIPTS / "september2026_topdown_categories.csv", encoding="utf-8-sig")
    td.columns = ["분류", "8월실제", "9월예측", "전월대비"]
    td_total = td.loc[td["분류"] == "총지수", "9월예측"].iloc[0]

    from hybrid_category_model import hybrid_total
    hyb_total = hybrid_total(hybrid)

    actual_total = ACTUAL_SEP["0 총지수"]
    total_row = pd.DataFrame([{
        "분류": "0 총지수", "바텀업예측": bu_total, "탑다운예측": td_total, "채택방식": "-",
        "최종예측": hyb_total, "가중치": 1000.0,
        "실제(9월)": actual_total, "실제(8월)": ACTUAL_AUG["0 총지수"],
        "실제_전월대비%": (actual_total - ACTUAL_AUG["0 총지수"]) / ACTUAL_AUG["0 총지수"] * 100,
        "최종오차": hyb_total - actual_total, "최종오차%": (hyb_total - actual_total) / actual_total * 100,
        "바텀업오차%": (bu_total - actual_total) / actual_total * 100,
        "탑다운오차%": (td_total - actual_total) / actual_total * 100,
    }])

    out = pd.concat([total_row, hybrid], ignore_index=True, sort=False)
    cols = ["분류", "실제(8월)", "실제(9월)", "실제_전월대비%", "바텀업예측", "바텀업오차%",
            "탑다운예측", "탑다운오차%", "최종예측", "최종오차%", "채택방식"]
    out = out[cols]
    out.to_csv(SCRIPTS / "september2026_actual_vs_pred_category.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))

    cats = out[out["분류"] != "0 총지수"]
    print("\n대분류(12) MAE: 바텀업", round(cats["바텀업오차%"].abs().mean(), 4),
          " 탑다운", round(cats["탑다운오차%"].abs().mean(), 4),
          " 최종(하이브리드)", round(cats["최종오차%"].abs().mean(), 4))
    print("\n오차 큰 순(최종채택 기준):")
    print(cats.reindex(cats["최종오차%"].abs().sort_values(ascending=False).index)[["분류", "채택방식", "최종오차%"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
