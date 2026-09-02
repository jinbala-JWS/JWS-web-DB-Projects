"""
2026년 8월 CPI 실제 발표치(KOSIS, 2026-09-02 확인) vs 우리 예측치 오차 검증.

실제값 출처: KOSIS 국가통계포털 "지출목적별 소비자물가지수(품목포함, 2020=100)"
(orgId=101, tblId=DT_1J22001) 검색결과 카드 + 통계표(html.do), 2026-09-02 조회.
현재 KOSIS 사이트의 레거시 트리 UI 제약으로 소분류(38)/세부품목(458) 실제치는
이번 세션에서 자동 추출하지 못해 총지수+대분류(12) 레벨만 실제-예측 비교가 가능하다.
세부품목 단위 "오차 큰 것" 분석은 대신 1년 워크포워드 백테스트(all_tiers_item_accuracy.csv)의
과거 MAPE를 프록시로 사용 - 8월 실측 개별품목 오차가 아니라 "이 품목의 예측이 원래
얼마나 불안정했는지"를 보여주는 것이므로 성격이 다름을 명확히 구분해서 표기한다.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# KOSIS 실제 발표치, 2026-08 (2026-09-02 조회)
ACTUAL_AUG = {
    "0 총지수": 120.05,
    "01 식료품 및 비주류음료": 128.27,
    "02 주류 및 담배": 105.29,
    "03 의류 및 신발": 119.72,
    "04 주택, 수도, 전기 및 연료": 117.59,
    "05 가정용품 및 가사 서비스": 123.16,
    "06 보건": 107.04,
    "07 교통": 124.63,
    "08 통신": 102.48,
    "09 오락 및 문화": 115.80,
    "10 교육": 110.31,
    "11 음식 및 숙박": 129.03,
    "12 기타 상품 및 서비스": 130.45,
}
ACTUAL_JUL = {
    "0 총지수": 119.77,
    "01 식료품 및 비주류음료": 126.94,
    "02 주류 및 담배": 105.09,
    "03 의류 및 신발": 119.71,
    "04 주택, 수도, 전기 및 연료": 117.29,
    "05 가정용품 및 가사 서비스": 122.90,
    "06 보건": 107.16,
    "07 교통": 124.73,
    "08 통신": 102.01,
    "09 오락 및 문화": 116.27,
    "10 교육": 110.27,
    "11 음식 및 숙박": 128.61,
    "12 기타 상품 및 서비스": 130.44,
}
LETTER_TO_CAT = {
    "A": "01 식료품 및 비주류음료", "B": "02 주류 및 담배", "C": "03 의류 및 신발",
    "D": "04 주택, 수도, 전기 및 연료", "E": "05 가정용품 및 가사 서비스", "F": "06 보건",
    "G": "07 교통", "H": "08 통신", "I": "09 오락 및 문화", "J": "10 교육",
    "K": "11 음식 및 숙박", "L": "12 기타 상품 및 서비스",
}


def blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def main():
    # 1) 탑다운(자체 히스토리 계절ETS) 대분류 예측
    topdown = pd.read_csv(SCRIPTS / "august2026_topdown_categories.csv", encoding="utf-8-sig")
    topdown.columns = ["분류", "2026-07(실제)", "2026-08(예측)", "전월대비%"]
    topdown_major = topdown[topdown["분류"].str.match(r"^\d\d ")].copy()

    # 2) 바텀업(458개 품목, R-ONE 전월세 반영 최종본)을 대분류로 집계
    bu = pd.read_csv(SCRIPTS / "august2026_bottomup_items_roneupdate.csv", encoding="utf-8-sig")
    bu.columns = ["품목코드", "품목명", "Tier", "가중치", "모델", "pred", "보정치레벨", "pred_corr", "pred_final"]
    bu["대분류코드"] = bu["품목코드"].str[0]
    bu["대분류"] = bu["대분류코드"].map(LETTER_TO_CAT)

    rows = []
    for cat, g in bu.groupby("대분류"):
        pred_bu = blend(g["pred_final"].values, g["가중치"].values)
        rows.append({"분류": cat, "바텀업_가중치합": g["가중치"].sum(), "바텀업예측": pred_bu})
    bu_cat = pd.DataFrame(rows)

    # 총지수(바텀업 전체)
    bu_total = blend(bu["pred_final"].values, bu["가중치"].values)

    # 3) 병합 + 오차 계산
    out = topdown_major.merge(bu_cat, on="분류", how="left")
    out["실제(8월)"] = out["분류"].map(ACTUAL_AUG)
    out["실제(7월)"] = out["분류"].map(ACTUAL_JUL)
    out["탑다운_오차"] = out["2026-08(예측)"] - out["실제(8월)"]
    out["탑다운_오차%"] = out["탑다운_오차"] / out["실제(8월)"] * 100
    out["바텀업_오차"] = out["바텀업예측"] - out["실제(8월)"]
    out["바텀업_오차%"] = out["바텀업_오차"] / out["실제(8월)"] * 100
    out["실제_전월대비%"] = (out["실제(8월)"] - out["실제(7월)"]) / out["실제(7월)"] * 100

    total_row = pd.DataFrame([{
        "분류": "0 총지수", "2026-08(예측)": topdown.loc[topdown["분류"] == "총지수", "2026-08(예측)"].iloc[0],
        "바텀업예측": bu_total,
        "실제(8월)": ACTUAL_AUG["0 총지수"], "실제(7월)": ACTUAL_JUL["0 총지수"],
    }])
    total_row["탑다운_오차"] = total_row["2026-08(예측)"] - total_row["실제(8월)"]
    total_row["탑다운_오차%"] = total_row["탑다운_오차"] / total_row["실제(8월)"] * 100
    total_row["바텀업_오차"] = total_row["바텀업예측"] - total_row["실제(8월)"]
    total_row["바텀업_오차%"] = total_row["바텀업_오차"] / total_row["실제(8월)"] * 100
    total_row["실제_전월대비%"] = (total_row["실제(8월)"] - total_row["실제(7월)"]) / total_row["실제(7월)"] * 100

    final = pd.concat([total_row, out[out["분류"] != "0 총지수"]], ignore_index=True, sort=False)
    cols = ["분류", "실제(7월)", "실제(8월)", "실제_전월대비%", "2026-08(예측)", "탑다운_오차", "탑다운_오차%",
            "바텀업예측", "바텀업_오차", "바텀업_오차%"]
    final = final[cols]
    final.to_csv(SCRIPTS / "august2026_actual_vs_pred_category.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print(final.round(3).to_string(index=False))

    print("\n탑다운 MAE(대분류12, 절대오차 평균):", out["탑다운_오차"].abs().mean().round(4))
    print("바텀업 MAE(대분류12, 절대오차 평균):", out["바텀업_오차"].abs().mean().round(4))
    print("\n오차 큰 순(탑다운 기준, 절대값):")
    print(out.reindex(out["탑다운_오차"].abs().sort_values(ascending=False).index)[["분류", "탑다운_오차", "탑다운_오차%"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
