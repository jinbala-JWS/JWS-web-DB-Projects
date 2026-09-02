"""
사용자가 KOSIS에서 직접 다운받은 전체 품목 실제 발표치(2026.08, 581행: 총지수1+대분류12+
소분류39+세분류71+세부품목458)를 파싱해, 세부품목(458)/소분류(39)/대분류(12)+총지수
전 레벨에서 예측 대비 오차를 계산한다.

원본: C:\\Users\\infomax\\Documents\\DB for Claude\\지출목적별_소비자물가지수_품목포함__2020100__20260902150814.xlsx
(사용자가 KOSIS 통계표에서 직접 다운로드 — evaluate_august_actual.py 작성 시점에는
자동화로 못 받았던 소분류/세부품목 레벨까지 전부 포함)
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

with open(SCRIPTS / "kosis_august_actual_full.json", encoding="utf-8") as f:
    raw = json.load(f)
actual_df = pd.DataFrame(raw)  # level(0~3), name, value

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
    # ---------- 1) 세부품목(458) ----------
    leaf_actual = actual_df[actual_df["level"] == 3][["name", "value"]].rename(
        columns={"name": "품목명", "value": "실제(8월)"})

    bu = pd.read_csv(SCRIPTS / "august2026_bottomup_items_roneupdate.csv", encoding="utf-8-sig")
    bu.columns = ["품목코드", "품목명", "Tier", "가중치", "모델", "pred_raw", "보정치레벨", "pred_corr", "예측(8월)"]

    m = bu.merge(leaf_actual, on="품목명", how="left")
    n_unmatched = m["실제(8월)"].isna().sum()

    m["오차"] = m["예측(8월)"] - m["실제(8월)"]
    m["오차%"] = m["오차"] / m["실제(8월)"] * 100
    m["대분류"] = m["품목코드"].str[0].map(LETTER_TO_CAT)
    m.to_csv(SCRIPTS / "august2026_leaf_actual_vs_pred.csv", index=False, encoding="utf-8-sig")

    matched = m.dropna(subset=["실제(8월)"])
    mae = matched["오차"].abs().mean()
    mape = matched["오차%"].abs().mean()
    wmape = np.average(matched["오차%"].abs(), weights=matched["가중치"])

    print(f"[세부품목 458] 매칭 {len(matched)}/{len(m)} (미매칭 {n_unmatched})")
    print(f"  MAE={mae:.4f}  MAPE(단순평균)={mape:.4f}%  가중MAPE={wmape:.4f}%")

    # ---------- 2) 소분류(39) ----------
    sub_actual = actual_df[actual_df["level"] == 1][["name", "value"]].rename(
        columns={"name": "분류", "value": "실제(8월)_소분류"})
    topdown = pd.read_csv(SCRIPTS / "august2026_topdown_categories.csv", encoding="utf-8-sig")
    topdown.columns = ["분류", "2026-07(실제)", "예측(8월)", "전월대비%"]
    sub_pred = topdown[topdown["분류"].str.match(r"^\d\d\.\d ")].copy()
    sub_m = sub_pred.merge(sub_actual, on="분류", how="left")
    sub_m["오차"] = sub_m["예측(8월)"] - sub_m["실제(8월)_소분류"]
    sub_m["오차%"] = sub_m["오차"] / sub_m["실제(8월)_소분류"] * 100
    sub_m.to_csv(SCRIPTS / "august2026_subcat_actual_vs_pred.csv", index=False, encoding="utf-8-sig")
    sub_matched = sub_m.dropna(subset=["실제(8월)_소분류"])
    print(f"\n[소분류 39] 매칭 {len(sub_matched)}/{len(sub_m)}")
    print(f"  MAE={sub_matched['오차'].abs().mean():.4f}  MAPE={sub_matched['오차%'].abs().mean():.4f}%")

    # ---------- 3) 대분류(12)+총지수 재확인 (기존 evaluate_august_actual.py와 교차검증) ----------
    major_actual = actual_df[actual_df["level"] == 0][["name", "value"]]
    print("\n[대분류+총지수] xlsx 실제치 (기존 웹조회 결과와 교차검증용):")
    print(major_actual.to_string(index=False))

    return m, sub_m


if __name__ == "__main__":
    main()
