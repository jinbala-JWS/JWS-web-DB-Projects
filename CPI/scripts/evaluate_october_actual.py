"""
2026년 10월 CPI 실제 발표치(KOSIS OpenAPI, kosis_cpi_202610.csv) vs 월말 확정 예측
(october2026_hybrid_category.csv / october2026_bottomup_items_roneupdate.csv /
october2026_topdown_categories.csv) 오차 검증.

1) 총지수 + 대분류(12): 바텀업/탑다운/하이브리드(공식) vs 실제
2) 세부품목(458): 오차, 총지수 기여오차, 오차 큰 항목
3) 실제(외부) 데이터를 쓰는 항목(오피넷 5 + R-ONE 전세·월세): 데이터를 안 썼다면 어땠을지(ETS)와 비교
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

import forecast_september_2026 as fs

SCRIPTS = Path(__file__).resolve().parent
norm = lambda x: re.sub(r"[\s,]+", "", str(x))
MONTH, PREV = "2026-10", "2026-09"

LETTER_TO_CAT = {
    "A": "01 식료품 및 비주류음료", "B": "02 주류 및 담배", "C": "03 의류 및 신발",
    "D": "04 주택, 수도, 전기 및 연료", "E": "05 가정용품 및 가사 서비스", "F": "06 보건",
    "G": "07 교통", "H": "08 통신", "I": "09 오락 및 문화", "J": "10 교육",
    "K": "11 음식 및 숙박", "L": "12 기타 상품 및 서비스",
}
REAL_DATA_ITEMS = {  # 품목 -> 어떤 실제 데이터를 쓰는가
    "휘발유": "오피넷 10월 31일 평균(회귀)", "경유": "오피넷 10월 31일 평균(회귀)",
    "등유": "오피넷 10월 31일 평균(회귀)", "자동차용LPG": "오피넷 10월 31일 평균(회귀)",
    "취사용LPG": "10월 오피넷 가격 미확보 -> 드리프트 폴백",
    "전세": "R-ONE 9월 지수 1개월시차 회귀", "월세": "R-ONE 9월 지수 1개월시차 회귀",
}


def main():
    act = pd.read_csv(SCRIPTS / "kosis_cpi_202610.csv", encoding="utf-8-sig")
    act["key"] = act["품목명"].map(norm)
    a_by = {lv: dict(zip(g["key"], g["지수"])) for lv, g in act.groupby("레벨")}
    prev_all = pd.read_csv(SCRIPTS / "kosis_cpi_202609.csv", encoding="utf-8-sig")
    prev_all["key"] = prev_all["품목명"].map(norm)
    p_by = {lv: dict(zip(g["key"], g["지수"])) for lv, g in prev_all.groupby("레벨")}

    # ---------- 1) 총지수 + 대분류 ----------
    hy = pd.read_csv(SCRIPTS / "october2026_hybrid_category.csv", encoding="utf-8-sig")
    hy["실제(10월)"] = hy["분류"].map(norm).map(a_by[1])
    hy["실제(9월)"] = hy["분류"].map(norm).map(p_by[1])
    hy["실제전월비%"] = (hy["실제(10월)"] / hy["실제(9월)"] - 1) * 100
    hy["예측전월비%"] = (hy["최종예측"] / hy["실제(9월)"] - 1) * 100
    for c_pred, c_err in [("바텀업예측", "바텀업오차%"), ("탑다운예측", "탑다운오차%"), ("최종예측", "최종오차%")]:
        hy[c_err] = (hy[c_pred] / hy["실제(10월)"] - 1) * 100
    hy["전년비_실제%"] = (hy["실제(10월)"] / hy["작년10월"] - 1) * 100
    hy["전년비_예측%"] = (hy["최종예측"] / hy["작년10월"] - 1) * 100
    cat_cols = ["분류", "채택방식", "실제(9월)", "실제(10월)", "실제전월비%", "예측전월비%", "바텀업예측", "바텀업오차%",
                "탑다운예측", "탑다운오차%", "최종예측", "최종오차%", "전년비_실제%", "전년비_예측%", "가중치"]
    hy[cat_cols].to_csv(SCRIPTS / "october2026_actual_vs_pred_category.csv", index=False, encoding="utf-8-sig")

    ts = pd.read_csv(SCRIPTS / "october2026_total_summary.csv", encoding="utf-8-sig")
    ts.columns = ["방식", "예측", "9월실제", "전월비%", "작년10월", "전년비%"]
    a_tot, p_tot = a_by[0]["0총지수"], p_by[0]["0총지수"]
    ts["실제"] = a_tot
    ts["오차%"] = (ts["예측"] / a_tot - 1) * 100
    ts["실제전월비%"] = (a_tot / p_tot - 1) * 100
    ts["실제전년비%"] = (a_tot / ts["작년10월"] - 1) * 100
    ts.to_csv(SCRIPTS / "october2026_actual_vs_pred_total.csv", index=False, encoding="utf-8-sig")
    print(ts.round(3).to_string(index=False))
    print(hy[["분류", "채택방식", "실제전월비%", "예측전월비%", "바텀업오차%", "탑다운오차%", "최종오차%"]].round(3).to_string(index=False))
    print("대분류 MAE(%): 바텀업", round(hy["바텀업오차%"].abs().mean(), 3), "탑다운", round(hy["탑다운오차%"].abs().mean(), 3),
          "최종", round(hy["최종오차%"].abs().mean(), 3))

    # ---------- 2) 세부품목 ----------
    bu = pd.read_csv(SCRIPTS / "october2026_bottomup_items_roneupdate.csv", encoding="utf-8-sig")
    bu.columns = ["품목코드", "품목명", "Tier", "가중치", "pred_raw", "보정치", "pred_corr", "pred_final"]
    bu["key"] = bu["품목명"].map(norm)
    bu["실제(9월)"] = bu["key"].map(p_by[4])
    bu["실제(10월)"] = bu["key"].map(a_by[4])
    bu["예측(10월)"] = bu["pred_final"]
    bu["오차"] = bu["예측(10월)"] - bu["실제(10월)"]
    bu["오차%"] = bu["오차"] / bu["실제(10월)"] * 100
    bu["기여오차"] = bu["가중치"] / 1000 * bu["오차"]
    bu["실제전월비%"] = (bu["실제(10월)"] / bu["실제(9월)"] - 1) * 100
    bu["예측전월비%"] = (bu["예측(10월)"] / bu["실제(9월)"] - 1) * 100
    bu["대분류"] = bu["품목코드"].str[0].map(LETTER_TO_CAT)
    bu["방향일치"] = np.sign(bu["실제전월비%"].round(2)) == np.sign(bu["예측전월비%"].round(2))
    leaf_cols = ["품목코드", "품목명", "Tier", "가중치", "대분류", "실제(9월)", "예측(10월)", "실제(10월)", "오차", "오차%",
                 "기여오차", "실제전월비%", "예측전월비%", "방향일치"]
    bu[leaf_cols].to_csv(SCRIPTS / "october2026_leaf_actual_vs_pred.csv", index=False, encoding="utf-8-sig")
    ok = bu.dropna(subset=["실제(10월)"])
    naive = (ok["실제(9월)"] / ok["실제(10월)"] - 1) * 100
    print(f"\n세부품목 매칭 {len(ok)}/{len(bu)}  MAE={ok['오차'].abs().mean():.3f}  MAPE={ok['오차%'].abs().mean():.3f}%  "
          f"가중MAPE={np.average(ok['오차%'].abs(), weights=ok['가중치']):.3f}%  "
          f"(참고: 전월값 그대로 가중MAPE={np.average(naive.abs(), weights=ok['가중치']):.3f}%)")
    print("기여오차 합:", round(ok["기여오차"].sum(), 4))
    moved = ok[ok["실제전월비%"].abs() >= 0.5]
    print(f"실제 ±0.5% 이상 움직인 {len(moved)}개 중 방향 일치 {moved['방향일치'].mean()*100:.1f}%")

    # ---------- 소분류(탑다운) ----------
    td = pd.read_csv(SCRIPTS / "october2026_topdown_categories.csv", encoding="utf-8-sig")
    td.columns = ["분류", "9월실제", "10월예측", "전월대비"]
    sub = td[td["분류"].str.match(r"^\d\d\.\d ")].copy()
    sub["실제(10월)"] = sub["분류"].map(norm).map(a_by[2])
    sub["오차%"] = (sub["10월예측"] / sub["실제(10월)"] - 1) * 100
    sub.to_csv(SCRIPTS / "october2026_subcat_actual_vs_pred.csv", index=False, encoding="utf-8-sig")
    print(f"소분류 {sub['실제(10월)'].notna().sum()}/{len(sub)}  탑다운 MAPE={sub['오차%'].abs().mean():.3f}%")

    # ---------- 3) 실제 데이터를 쓰는 항목 ----------
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv", encoding="utf-8-sig")
    panel.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"] + list(panel.columns[6:])
    month_cols = sorted([c for c in panel.columns if c[:2] in ("19", "20")])
    train = [c for c in month_cols if c < MONTH]
    rows = []
    for name, src in REAL_DATA_ITEMS.items():
        r = bu[bu["품목명"] == name].iloc[0]
        ser = pd.Series(panel[panel["품목명"] == name].iloc[0][month_cols].astype(float).values, index=month_cols)
        hist = ser[train].interpolate(limit_area="inside")
        ets = fs.ets_forecast(hist, False)
        own = r["pred_corr"] if name in ("전세", "월세") else r["pred_final"]
        actual = r["실제(10월)"]
        rows.append({
            "품목": name, "가중치": r["가중치"], "사용 실제데이터": src,
            "9월실제": r["실제(9월)"], "10월실제": actual, "실제전월비%": r["실제전월비%"],
            "모델예측(최종)": r["pred_final"], "모델오차%": (r["pred_final"] / actual - 1) * 100,
            "데이터없이_비계절ETS예측": ets, "ETS오차%": (ets / actual - 1) * 100,
            "전월값그대로_오차%": (r["실제(9월)"] / actual - 1) * 100,
            "자체모델(보정후)예측": own, "자체모델오차%": (own / actual - 1) * 100,
        })
    real = pd.DataFrame(rows)
    real.to_csv(SCRIPTS / "october2026_realdata_items_check.csv", index=False, encoding="utf-8-sig")
    print(real.round(3).to_string(index=False))
    w = real["가중치"]
    print("가중MAPE: 모델", round(np.average(real["모델오차%"].abs(), weights=w), 3), "ETS", round(np.average(real["ETS오차%"].abs(), weights=w), 3),
          "전월그대로", round(np.average(real["전월값그대로_오차%"].abs(), weights=w), 3))


if __name__ == "__main__":
    main()
