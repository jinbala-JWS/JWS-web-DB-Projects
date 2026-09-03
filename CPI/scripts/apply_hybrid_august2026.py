"""
hybrid_category_model.py를 8월 예측치에 적용 — 이미 실제 발표치가 나온 8월을 대상으로
"하이브리드를 처음부터 공식 예측으로 채택했다면 얼마나 더 정확했을까"를 검증한다.
(이 스크립트 자체가 모델 변경의 최초 적용 사례이자 회고적 검증)

바텀업 소스: august2026_bottomup_items_roneupdate.csv (게이트 편향보정 + R-ONE
전월세 반영까지 끝난, 이번 8월 예측의 최종 바텀업 leaf 예측치)
탑다운 소스: august2026_topdown_categories.csv (대분류 12개 행만 사용)
"""
import pandas as pd
from pathlib import Path
from hybrid_category_model import bottomup_to_category, combine_hybrid, hybrid_total

SCRIPTS = Path(__file__).resolve().parent

ACTUAL_AUG = {
    "0 총지수": 120.05,
    "01 식료품 및 비주류음료": 128.27, "02 주류 및 담배": 105.29, "03 의류 및 신발": 119.72,
    "04 주택, 수도, 전기 및 연료": 117.59, "05 가정용품 및 가사 서비스": 123.16, "06 보건": 107.04,
    "07 교통": 124.63, "08 통신": 102.48, "09 오락 및 문화": 115.80, "10 교육": 110.31,
    "11 음식 및 숙박": 129.03, "12 기타 상품 및 서비스": 130.45,
}


def main():
    bu_items = pd.read_csv(SCRIPTS / "august2026_bottomup_items_roneupdate.csv", encoding="utf-8-sig")
    bu_items.columns = ["품목코드", "품목명", "Tier", "가중치", "모델", "pred_raw", "보정치레벨", "pred_corr", "pred_final"]
    bu_cat = bottomup_to_category(bu_items, pred_col="pred_final")

    td = pd.read_csv(SCRIPTS / "august2026_topdown_categories.csv", encoding="utf-8-sig")
    td.columns = ["분류", "7월(실제)", "탑다운예측", "전월대비%"]
    td_major = td[td["분류"].str.match(r"^\d\d ")][["분류", "탑다운예측"]]

    hybrid = combine_hybrid(bu_cat, td_major)
    hybrid["실제(8월)"] = hybrid["분류"].map(ACTUAL_AUG)
    hybrid["오차"] = hybrid["최종예측"] - hybrid["실제(8월)"]
    hybrid["오차%"] = hybrid["오차"] / hybrid["실제(8월)"] * 100
    hybrid = hybrid.sort_values("분류")
    hybrid.to_csv(SCRIPTS / "august2026_hybrid_category.csv", index=False, encoding="utf-8-sig")

    total_hybrid = hybrid_total(hybrid)
    total_actual = ACTUAL_AUG["0 총지수"]

    print(hybrid[["분류", "바텀업예측", "탑다운예측", "채택방식", "최종예측", "실제(8월)", "오차%"]].round(4).to_string(index=False))

    mae = hybrid["오차"].abs().mean()
    mape = hybrid["오차%"].abs().mean()
    print(f"\n하이브리드 대분류(12) MAE={mae:.4f}  MAPE={mape:.4f}%")
    print(f"하이브리드 총지수 추정: {total_hybrid:.4f}  (실제 {total_actual}, 오차 {total_hybrid - total_actual:+.4f} / {(total_hybrid-total_actual)/total_actual*100:+.4f}%)")

    # 기존(순수바텀업/순수탑다운) 대비 비교
    pure_bu_mape = (bu_cat.merge(pd.Series(ACTUAL_AUG, name="실제").reset_index().rename(columns={"index": "분류"}), on="분류")
                     .assign(오차퍼센트=lambda d: (d["바텀업예측"] - d["실제"]) / d["실제"] * 100))["오차퍼센트"].abs().mean()
    pure_td_mape = (td_major.merge(pd.Series(ACTUAL_AUG, name="실제").reset_index().rename(columns={"index": "분류"}), on="분류")
                     .assign(오차퍼센트=lambda d: (d["탑다운예측"] - d["실제"]) / d["실제"] * 100))["오차퍼센트"].abs().mean()
    print(f"\n[참고] 순수바텀업 MAPE={pure_bu_mape:.4f}%  순수탑다운 MAPE={pure_td_mape:.4f}%  하이브리드 MAPE={mape:.4f}%")


if __name__ == "__main__":
    main()
