"""
대분류(12)별로 "탑다운이 확실히 더 정확한 카테고리"만 탑다운을, 나머지는 바텀업을
쓰는 하이브리드 방식을 구성하고, 24개월(2024-08~2026-07) 워크포워드로 재검증한다.

"확실히"의 기준: 24개월간 카테고리별 (바텀업 절대오차 - 탑다운 절대오차) 쌍체 표본에
Wilcoxon signed-rank 검정(비모수, 이상치에 덜 민감 - 07 교통처럼 한두 달 튀는 값이
있어도 안정적)을 적용해 p<0.05이고 탑다운이 더 정확한 방향일 때만 "확실히 탑다운
우위"로 채택한다. 그 외(유의차 없음 + 바텀업이 유의하게 우위인 경우 전부)는 바텀업을 쓴다.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

SCRIPTS = Path(__file__).resolve().parent
ALPHA = 0.05


def classify():
    df = pd.read_csv(SCRIPTS / "topdown_vs_bottomup_category_backtest.csv", encoding="utf-8-sig")
    df.columns = ["월", "분류", "방식", "예측", "실제", "오차", "오차%"]
    df["절대오차"] = df["오차"].abs()

    wide = df.pivot_table(index=["분류", "월"], columns="방식", values="절대오차").reset_index()

    rows = []
    for cat, g in wide.groupby("분류"):
        g = g.dropna(subset=["바텀업", "탑다운"])
        diff = g["바텀업"] - g["탑다운"]  # 양수 = 탑다운이 더 정확했던 달
        if (diff != 0).sum() < 5:
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(diff)
        mean_bu, mean_td = g["바텀업"].mean(), g["탑다운"].mean()
        topdown_better = mean_td < mean_bu
        significant = p < ALPHA and topdown_better
        rows.append({
            "분류": cat, "n": len(g), "바텀업_평균절대오차": mean_bu, "탑다운_평균절대오차": mean_td,
            "탑다운이_더정확한_달수": int((diff > 0).sum()), "Wilcoxon_p": p,
            "확실히_탑다운우위": significant,
            "채택방식": "탑다운" if significant else "바텀업",
        })
    result = pd.DataFrame(rows).sort_values("Wilcoxon_p")
    result.to_csv(SCRIPTS / "category_method_selection.csv", index=False, encoding="utf-8-sig")
    print(result.round(4).to_string(index=False))
    return result


def backtest_hybrid(selection: pd.DataFrame):
    df = pd.read_csv(SCRIPTS / "topdown_vs_bottomup_category_backtest.csv", encoding="utf-8-sig")
    df.columns = ["월", "분류", "방식", "예측", "실제", "오차", "오차%"]

    method_map = dict(zip(selection["분류"], selection["채택방식"]))
    df["채택"] = df["분류"].map(method_map)
    hybrid = df[df["방식"] == df["채택"]].copy()

    # 대분류 가중치(카테고리 자체 비교이므로 CPI_Tier_분류.csv 각 대분류 총가중치 사용)
    tier = pd.read_csv(SCRIPTS.parent / "CPI_Tier_분류.csv", encoding="utf-8-sig")
    tier.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"]
    LETTER_TO_CAT = {
        "A": "01 식료품 및 비주류음료", "B": "02 주류 및 담배", "C": "03 의류 및 신발",
        "D": "04 주택, 수도, 전기 및 연료", "E": "05 가정용품 및 가사 서비스", "F": "06 보건",
        "G": "07 교통", "H": "08 통신", "I": "09 오락 및 문화", "J": "10 교육",
        "K": "11 음식 및 숙박", "L": "12 기타 상품 및 서비스",
    }
    tier["대분류"] = tier["품목코드"].str[0].map(LETTER_TO_CAT)
    cat_weight = tier.groupby("대분류")["가중치"].sum()

    hybrid["가중치"] = hybrid["분류"].map(cat_weight)
    hybrid.to_csv(SCRIPTS / "hybrid_category_backtest.csv", index=False, encoding="utf-8-sig")

    # ---- 카테고리별 요약: 하이브리드가 실제로 그 카테고리에서만 봤을 때 개선됐는지 ----
    def mae_mape(g):
        return pd.Series({"MAE": g["오차"].abs().mean(), "MAPE": g["오차%"].abs().mean(), "n": len(g)})

    per_cat = hybrid.groupby("분류").apply(mae_mape).reset_index()
    per_cat = per_cat.merge(selection[["분류", "채택방식"]], on="분류")
    per_cat.to_csv(SCRIPTS / "hybrid_category_summary.csv", index=False, encoding="utf-8-sig")
    print("\n[하이브리드 채택 후 카테고리별 정확도]")
    print(per_cat.round(4).to_string(index=False))

    # ---- 전체(가중평균) 비교: 순수 탑다운 vs 순수 바텀업 vs 하이브리드 ----
    def weighted_mape(sub_df):
        return np.average(sub_df["오차%"].abs(), weights=sub_df["가중치"])

    pure_td = df[df["방식"] == "탑다운"].copy()
    pure_bu = df[df["방식"] == "바텀업"].copy()
    pure_td["가중치"] = pure_td["분류"].map(cat_weight)
    pure_bu["가중치"] = pure_bu["분류"].map(cat_weight)

    print("\n[12개 대분류 전체 가중MAPE 비교 (24개월 평균)]")
    print(f"  순수 탑다운만 사용: {weighted_mape(pure_td):.4f}%")
    print(f"  순수 바텀업만 사용: {weighted_mape(pure_bu):.4f}%")
    print(f"  하이브리드(카테고리별 확실한 쪽 채택): {weighted_mape(hybrid):.4f}%")

    # 월별 가중MAPE도 비교(월별 변동 확인용)
    monthly = []
    for m, g in hybrid.groupby("월"):
        monthly.append({"월": m, "방식": "하이브리드", "가중MAPE": weighted_mape(g)})
    for m, g in pure_td.groupby("월"):
        monthly.append({"월": m, "방식": "순수탑다운", "가중MAPE": weighted_mape(g)})
    for m, g in pure_bu.groupby("월"):
        monthly.append({"월": m, "방식": "순수바텀업", "가중MAPE": weighted_mape(g)})
    monthly_df = pd.DataFrame(monthly)
    monthly_df.to_csv(SCRIPTS / "hybrid_monthly_comparison.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    sel = classify()
    backtest_hybrid(sel)
