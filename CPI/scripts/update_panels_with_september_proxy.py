"""
10월 예측을 위한 선행작업 — 8월판(update_panels_with_august_actual.py)과 같은 역할이지만,
9월은 아직 KOSIS 세부품목(458/581개) 레벨 실제 발표치를 확보하지 못한 상태(evaluate_september_actual.py는
대분류 12개+총지수만 검증함 - 사용자가 지난달처럼 xlsx를 제공하면 교체 가능)라 **프록시(proxy)**로 채운다.

- 0 총지수 + 대분류(12): 실제 발표치 사용(evaluate_september_actual.py의 ACTUAL_SEP, KOSIS 확인 2026-10-02)
- 소분류(38): 실제값 없음 → september2026_topdown_categories.csv의 예측치(탑다운 ETS)로 대체
- 458개 leaf 품목(all_tiers_monthly_panel.csv): 실제값 없음 → september2026_bottomup_items_corrected.csv의
  '보정후_pred_2026-09'(바텀업+게이트 편향보정)로 대체

→ 즉 10월 예측은 "9월 실제치 100% 확정 기반"이 아니라 "9월 상위 13개 계열만 확정, 나머지는 9월
자체 예측치 기반"이라는 한계가 있다. 추후 KOSIS 9월 세부품목 xlsx가 확보되면 이 스크립트를
update_panels_with_september_actual.py로 교체하고 10월 예측을 재실행해야 한다.
"""
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

ACTUAL_SEP_MAJOR = {
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


def update_official():
    df = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv", encoding="utf-8-sig")
    topdown = pd.read_csv(SCRIPTS / "september2026_topdown_categories.csv", encoding="utf-8-sig")
    sub_proxy = dict(zip(topdown["분류"], topdown["2026-09(예측)"]))
    # 총지수/대분류는 topdown 파일에 "총지수"로만 표기되어 있어 라벨을 맞춰준다
    sub_proxy = {("0 총지수" if k == "총지수" else k): v for k, v in sub_proxy.items()}

    proxy_all = dict(sub_proxy)
    proxy_all.update(ACTUAL_SEP_MAJOR)  # 실제값이 있는 13개는 실제값으로 덮어씀(우선)

    df["2026-09"] = df["품목"].map(proxy_all)
    n_real = df["품목"].isin(ACTUAL_SEP_MAJOR.keys()).sum()
    n_proxy = df["품목"].isin(sub_proxy.keys()).sum() - n_real
    n_missing = df["2026-09"].isna().sum()
    df.to_csv(SCRIPTS / "cpi_official_monthly_wide.csv", index=False, encoding="utf-8-sig")
    print(f"[cpi_official_monthly_wide.csv] 실제값 {n_real}행(총지수+대분류), "
          f"프록시(탑다운예측) {n_proxy}행(소분류), 미매칭(leaf 등) {n_missing}행")
    return df


def update_all_tiers_panel():
    df = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv", encoding="utf-8-sig")
    bu = pd.read_csv(SCRIPTS / "september2026_bottomup_items_corrected.csv", encoding="utf-8-sig")
    proxy = dict(zip(bu["품목명"], bu["보정후_pred_2026-09"]))
    name_col = df.columns[1]
    df["2026-09"] = df[name_col].map(proxy)
    n_missing = df["2026-09"].isna().sum()
    df.to_csv(SCRIPTS / "all_tiers_monthly_panel.csv", index=False, encoding="utf-8-sig")
    print(f"[all_tiers_monthly_panel.csv] {len(df)}행 전부 프록시(바텀업+편향보정 예측치)로 반영, 미매칭 {n_missing}행")
    return df


if __name__ == "__main__":
    update_official()
    update_all_tiers_panel()
