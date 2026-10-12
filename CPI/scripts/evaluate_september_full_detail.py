"""
9월 CPI 세부품목(458)/소분류(39) 실제치(KOSIS OpenAPI, kosis_cpi_202609.csv) vs
최종 9월 예측(september2026_bottomup_items_corrected.csv, 소분류는 탑다운 csv) 오차.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
norm = lambda x: re.sub(r"[\s,]+", "", str(x))

LETTER_TO_CAT = {
    "A": "01 식료품 및 비주류음료", "B": "02 주류 및 담배", "C": "03 의류 및 신발",
    "D": "04 주택, 수도, 전기 및 연료", "E": "05 가정용품 및 가사 서비스", "F": "06 보건",
    "G": "07 교통", "H": "08 통신", "I": "09 오락 및 문화", "J": "10 교육",
    "K": "11 음식 및 숙박", "L": "12 기타 상품 및 서비스",
}


def main():
    act = pd.read_csv(SCRIPTS / "kosis_cpi_202609.csv", encoding="utf-8-sig")
    leaf = act[act["레벨"] == 4].copy()
    leaf["key"] = leaf["품목명"].map(norm)
    leaf_actual = dict(zip(leaf["key"], leaf["지수"]))

    bu = pd.read_csv(SCRIPTS / "september2026_bottomup_items_corrected.csv", encoding="utf-8-sig")
    bu["실제(9월)"] = bu["품목명"].map(norm).map(leaf_actual)
    bu["예측(9월)"] = bu["보정후_pred_2026-09"]
    bu["오차"] = bu["예측(9월)"] - bu["실제(9월)"]
    bu["오차%"] = bu["오차"] / bu["실제(9월)"] * 100
    bu["기여오차"] = bu["가중치"] / 1000 * bu["오차"]
    bu["대분류"] = bu["품목코드"].str[0].map(LETTER_TO_CAT)
    cols = ["품목코드", "품목명", "Tier", "가중치", "모델", "예측(9월)", "실제(9월)", "오차", "오차%", "기여오차", "대분류"]
    bu[cols].to_csv(SCRIPTS / "september2026_leaf_actual_vs_pred.csv", index=False, encoding="utf-8-sig")

    ok = bu.dropna(subset=["실제(9월)"])
    print(f"세부품목 매칭 {len(ok)}/{len(bu)}")
    print(f"MAE={ok['오차'].abs().mean():.3f}  MAPE={ok['오차%'].abs().mean():.3f}%  "
          f"가중MAPE={np.average(ok['오차%'].abs(), weights=ok['가중치']):.3f}%")
    print("기여오차 합:", round(ok["기여오차"].sum(), 4))

    # 소분류
    td = pd.read_csv(SCRIPTS / "september2026_topdown_categories.csv", encoding="utf-8-sig")
    td.columns = ["분류", "8월실제", "9월예측", "전월대비"]
    sub = td[td["분류"].str.match(r"^\d\d\.\d ")].copy()
    sub_act = {norm(n): v for n, v in zip(act[act["레벨"] == 2]["품목명"], act[act["레벨"] == 2]["지수"])}
    sub["실제(9월)"] = sub["분류"].map(norm).map(sub_act)
    sub["오차%"] = (sub["9월예측"] - sub["실제(9월)"]) / sub["실제(9월)"] * 100
    sub.to_csv(SCRIPTS / "september2026_subcat_actual_vs_pred.csv", index=False, encoding="utf-8-sig")
    print(f"소분류 매칭 {sub['실제(9월)'].notna().sum()}/{len(sub)}  MAPE={sub['오차%'].abs().mean():.3f}%")


if __name__ == "__main__":
    main()
