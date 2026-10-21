"""
04(주택,수도,전기및연료) 대분류를 다시 구성 — 04.1 주택임차료(전세·월세)는
R-ONE 시차회귀 수치로 직접 대입하고, 나머지 소분류(04.3 주거시설 유지보수,
04.4 수도 및 주거관련 서비스, 04.5 전기·가스및기타연료)만 탑다운(자체 히스토리
계절ETS)으로 예측해 가중평균으로 04 전체를 재구성한다.

기존 공식 하이브리드는 04 전체를 통째로 탑다운(119.441)으로 채택했는데, 이는
04.1(주택임차료=전세+월세)의 계절ETS 적합 오차와 04.3~04.5의 적합 오차가 섞여있다.
전세·월세는 R-ONE 외부지수 회귀모델이라는 독립적인 교차검증 수단이 있으므로,
이를 수치로 고정하고 나머지 3개 소분류만 탑다운을 적용하면 04 전체를 더 정교하게
구성할 수 있다는 아이디어.

실행: python october_cat04_rone_plus_topdown.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

import forecast_september_2026 as fs

SCRIPTS = Path(__file__).resolve().parent
TARGET = "2026-10"

RONE_JEONSE = 106.685
RONE_WOLSE = 106.340
W_JEONSE, W_WOLSE = 54.2, 44.9

SUBCATS = ["04.3 주거시설 유지 · 보수", "04.4 수도 및 주거관련 서비스", "04.5 전기, 가스 및 기타연료"]
SUBCAT_WEIGHTS = {"04.3 주거시설 유지 · 보수": 9.3, "04.4 수도 및 주거관련 서비스": 30.6,
                  "04.5 전기, 가스 및 기타연료": 32.6}
CAT04_NAME = "04 주택, 수도, 전기 및 연료"


def main():
    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    item_col = official.columns[0]
    official = official.drop_duplicates(subset=item_col, keep="first")
    month_cols = sorted([c for c in official.columns if c[:2] in ("19", "20")])
    train_cols = [c for c in month_cols if c < TARGET]

    # 1) 04.1 주택임차료 = 전세·월세 R-ONE 수치 가중평균(공식 Laspeyres 방식 = 가중산술평균)
    jeonse_wolse_val = (W_JEONSE * RONE_JEONSE + W_WOLSE * RONE_WOLSE) / (W_JEONSE + W_WOLSE)
    print(f"04.1 주택임차료(R-ONE 수치 대입) = {jeonse_wolse_val:.3f} "
          f"(전세 {RONE_JEONSE} x{W_JEONSE} + 월세 {RONE_WOLSE} x{W_WOLSE})")

    # 2) 04.3~04.5는 자체 히스토리 계절ETS(top_down_for_month와 동일 방식)
    sub_preds = {}
    for name in SUBCATS:
        row = official[official[item_col] == name].iloc[0]
        series = pd.Series({c: row[c] for c in month_cols}).astype(float)
        hist = series[train_cols].dropna()
        pred = fs.ets_forecast(hist, is_seasonal=True)
        sub_preds[name] = pred
        prev = series[train_cols[-1]]
        print(f"{name}: 탑다운예측 {pred:.3f} (9월실제 {prev:.2f}, 전월비 {(pred/prev-1)*100:+.3f}%)")

    # 3) 04 전체 재구성 = 가중산술평균(99.1 + 9.3+30.6+32.6 = 171.6)
    w_rent = W_JEONSE + W_WOLSE
    total_w = w_rent + sum(SUBCAT_WEIGHTS.values())
    cat04_new = (w_rent * jeonse_wolse_val + sum(sub_preds[n] * SUBCAT_WEIGHTS[n] for n in SUBCATS)) / total_w

    # 비교용: 기존 04 전체 탑다운(자체 히스토리로 04 전체를 직접 적합)
    row04 = official[official[item_col] == CAT04_NAME].iloc[0]
    series04 = pd.Series({c: row04[c] for c in month_cols}).astype(float)
    hist04 = series04[train_cols].dropna()
    cat04_pure_topdown = fs.ets_forecast(hist04, is_seasonal=True)
    prev04 = series04[train_cols[-1]]

    print(f"\n04 전체(가중치 {total_w}) 재구성값(주택임차료 수치대입+나머지 탑다운): {cat04_new:.3f} "
          f"(전월비 {(cat04_new/prev04-1)*100:+.3f}%)")
    print(f"04 전체 기존 순수탑다운(04 자체 히스토리 직접적합):           {cat04_pure_topdown:.3f} "
          f"(전월비 {(cat04_pure_topdown/prev04-1)*100:+.3f}%)")
    print(f"차이: {cat04_new - cat04_pure_topdown:+.3f}")

    # 4) 공식 하이브리드 총지수에서 04를 이 재구성값으로 교체했을 때 총지수 변화
    hybrid = pd.read_csv(SCRIPTS / "october2026_hybrid_category.csv", encoding="utf-8-sig")
    before_total = float(np.average(hybrid["최종예측"], weights=hybrid["가중치"]))
    hybrid2 = hybrid.copy()
    mask = hybrid2["분류"] == CAT04_NAME
    assert mask.sum() == 1, hybrid2["분류"].tolist()
    old04 = hybrid2.loc[mask, "최종예측"].iloc[0]
    hybrid2.loc[mask, "최종예측"] = cat04_new
    after_total = float(np.average(hybrid2["최종예측"], weights=hybrid2["가중치"]))

    print(f"\n[총지수 영향] 기존 04={old04:.3f} 적용 하이브리드 총지수: {before_total:.4f}")
    print(f"[총지수 영향] 신규 04={cat04_new:.3f} 적용 하이브리드 총지수: {after_total:.4f}")
    print(f"변화: {after_total - before_total:+.4f}")

    out = pd.DataFrame([
        {"구성요소": "04.1 주택임차료(R-ONE수치대입)", "가중치": w_rent, "값": jeonse_wolse_val},
        *[{"구성요소": n, "가중치": SUBCAT_WEIGHTS[n], "값": sub_preds[n]} for n in SUBCATS],
        {"구성요소": "04 전체(재구성)", "가중치": total_w, "값": cat04_new},
        {"구성요소": "04 전체(기존 순수탑다운)", "가중치": total_w, "값": cat04_pure_topdown},
        {"구성요소": "총지수(기존 04)", "가중치": None, "값": before_total},
        {"구성요소": "총지수(신규 04)", "가중치": None, "값": after_total},
    ])
    out.to_csv(SCRIPTS / "october2026_cat04_rone_plus_topdown.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
