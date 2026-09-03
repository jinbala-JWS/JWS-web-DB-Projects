"""
"카테고리(대분류12) 가중산술평균으로 재구성한 총지수" 방식을 고정하고, 카테고리
선택(순수탑다운/순수바텀업/하이브리드)만 바꿔가며 24개월 워크포워드로 총지수 자체의
정확도를 비교한다. hybrid_category_model.hybrid_total()의 정확도 주장을 뒷받침하는
근거 스크립트.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from hybrid_category_model import LETTER_TO_CAT, category_weights

SCRIPTS = Path(__file__).resolve().parent


def main():
    bt = pd.read_csv(SCRIPTS / "topdown_vs_bottomup_category_backtest.csv", encoding="utf-8-sig")
    bt.columns = ["월", "분류", "방식", "예측", "실제", "오차", "오차%"]

    sel = pd.read_csv(SCRIPTS / "category_method_selection.csv", encoding="utf-8-sig")
    sel_map = dict(zip(sel["분류"], sel["채택방식"]))

    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = official.columns[0]
    total_row = official[official[item_col] == "0 총지수"].iloc[0]

    w = category_weights()

    rows = []
    for m in sorted(bt["월"].unique()):
        sub = bt[bt["월"] == m]
        td = sub[sub["방식"] == "탑다운"].set_index("분류")["예측"]
        bu = sub[sub["방식"] == "바텀업"].set_index("분류")["예측"]
        hyb_vals = {cat: (td[cat] if sel_map.get(cat, "바텀업") == "탑다운" else bu[cat]) for cat in w.index}

        rows.append({
            "월": m,
            "실제총지수": total_row[m],
            "하이브리드총지수": np.average([hyb_vals[c] for c in w.index], weights=w.values),
            "탑다운총지수(카테고리재구성)": np.average([td[c] for c in w.index], weights=w.values),
            "바텀업총지수(카테고리재구성)": np.average([bu[c] for c in w.index], weights=w.values),
        })

    df = pd.DataFrame(rows)
    df["하이브리드오차%"] = (df["하이브리드총지수"] - df["실제총지수"]) / df["실제총지수"] * 100
    df["탑다운오차%"] = (df["탑다운총지수(카테고리재구성)"] - df["실제총지수"]) / df["실제총지수"] * 100
    df["바텀업오차%"] = (df["바텀업총지수(카테고리재구성)"] - df["실제총지수"]) / df["실제총지수"] * 100
    df.to_csv(SCRIPTS / "hybrid_total_backtest_24m.csv", index=False, encoding="utf-8-sig")

    print(f"하이브리드 총지수 MAPE(24개월): {df['하이브리드오차%'].abs().mean():.4f}%  (bias {df['하이브리드오차%'].mean():+.4f}%)")
    print(f"탑다운(카테고리재구성) 총지수 MAPE: {df['탑다운오차%'].abs().mean():.4f}%")
    print(f"바텀업(카테고리재구성) 총지수 MAPE: {df['바텀업오차%'].abs().mean():.4f}%")
    return df


if __name__ == "__main__":
    main()
