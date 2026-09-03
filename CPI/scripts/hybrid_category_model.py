"""
카테고리(대분류 12개) 레벨 하이브리드 결합 로직 — 표준 모듈.

[하이브리드_탑다운_바텀업_백테스트.md](../하이브리드_탑다운_바텀업_백테스트.md)에서
24개월 워크포워드 + Wilcoxon 검정(p<0.05)으로 검증한 결과, "확실히 탑다운이 더
정확한" 카테고리는 04(주택,수도,전기및연료)·06(보건)·09(오락및문화) 3개뿐이었고
(하이브리드 가중MAPE 0.392% vs 순수탑다운 0.436% vs 순수바텀업 0.473% — 하이브리드가
둘 다보다 우수), 나머지 카테고리는 바텀업이 같거나 더 나았다.

이 모듈은 그 결론을 코드화한 것이다. **앞으로의 모든 월별 예측(forecast_*_2026.py류
스크립트)은 카테고리/총지수 단계에서 "탑다운 단독"이나 "바텀업 단독"이 아니라 이
하이브리드 결합을 공식 예측치로 채택한다.**

채택 기준 자체는 [category_method_selection.csv](./category_method_selection.csv)에서
읽어온다 — 매달 데이터가 쌓이면 [backtest_hybrid_category.py](./backtest_hybrid_category.py)를
재실행해 이 CSV를 갱신하면 이 모듈은 자동으로 최신 판정을 반영한다(하드코딩 없음).
"""
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

LETTER_TO_CAT = {
    "A": "01 식료품 및 비주류음료", "B": "02 주류 및 담배", "C": "03 의류 및 신발",
    "D": "04 주택, 수도, 전기 및 연료", "E": "05 가정용품 및 가사 서비스", "F": "06 보건",
    "G": "07 교통", "H": "08 통신", "I": "09 오락 및 문화", "J": "10 교육",
    "K": "11 음식 및 숙박", "L": "12 기타 상품 및 서비스",
}


def load_hybrid_selection() -> dict:
    """카테고리별 채택방식({대분류명: '탑다운'|'바텀업'})을 백테스트 결과 CSV에서 읽는다."""
    sel = pd.read_csv(SCRIPTS / "category_method_selection.csv", encoding="utf-8-sig")
    return dict(zip(sel["분류"], sel["채택방식"]))


def category_weights() -> pd.Series:
    """CPI_Tier_분류.csv(458개 leaf, 품목코드 첫 글자=대분류)로부터 대분류별 총가중치 산출."""
    tier = pd.read_csv(SCRIPTS.parent / "CPI_Tier_분류.csv", encoding="utf-8-sig")
    tier.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"]
    tier["대분류"] = tier["품목코드"].str[0].map(LETTER_TO_CAT)
    return tier.groupby("대분류")["가중치"].sum()


def _blend(vals, weights):
    arith = np.average(vals, weights=weights)
    geom = np.exp(np.average(np.log(vals), weights=weights))
    return 0.5 * arith + 0.5 * geom


def bottomup_to_category(items_df: pd.DataFrame, pred_col: str, code_col: str = "품목코드",
                          weight_col: str = "가중치") -> pd.DataFrame:
    """458개 leaf 품목 예측치 DataFrame을 대분류(12)로 가중집계(산술+기하 50:50 블렌드)."""
    df = items_df.copy()
    df["대분류"] = df[code_col].str[0].map(LETTER_TO_CAT)
    rows = []
    for cat, g in df.groupby("대분류"):
        rows.append({"분류": cat, "바텀업예측": _blend(g[pred_col].values, g[weight_col].values)})
    return pd.DataFrame(rows)


def combine_hybrid(bottomup_by_cat: pd.DataFrame, topdown_by_cat: pd.DataFrame,
                    selection: dict = None) -> pd.DataFrame:
    """
    bottomup_by_cat: 컬럼 [분류, 바텀업예측] (12행, bottomup_to_category() 결과)
    topdown_by_cat: 컬럼 [분류, 탑다운예측] (12행, 대분류만 — 총지수/소분류 제외하고 넘길 것)
    selection: {대분류명: '탑다운'|'바텀업'}, None이면 category_method_selection.csv에서 로드

    반환: [분류, 바텀업예측, 탑다운예측, 채택방식, 최종예측] + 가중치 컬럼
    """
    if selection is None:
        selection = load_hybrid_selection()

    out = bottomup_by_cat.merge(topdown_by_cat, on="분류", how="outer")
    out["채택방식"] = out["분류"].map(selection).fillna("바텀업")  # 판정 없는 카테고리는 기본 바텀업(보수적)
    out["최종예측"] = np.where(out["채택방식"] == "탑다운", out["탑다운예측"], out["바텀업예측"])

    w = category_weights()
    out["가중치"] = out["분류"].map(w)
    return out


def hybrid_total(hybrid_df: pd.DataFrame) -> float:
    """카테고리별 최종예측을 가중치로 산술평균해 총지수 추정치 산출.
    (대분류 12개는 이미 leaf 458개의 Laspeyres형 집계 결과라, leaf 단계와 달리
    산술+기하 블렌드가 아니라 단순 가중산술평균을 쓴다.)

    **24개월(2024-08~2026-07) 워크포워드로 검증한 결과** — 이 "카테고리 가중산술평균"
    총지수 재구성 방식 자체를 고정하고 카테고리 선택만 바꿔봤을 때:
      - 하이브리드(선택적 채택) 총지수 MAPE = 0.124%
      - 순수 탑다운(12개 전부) 총지수 MAPE = 0.171%
      - 순수 바텀업(12개 전부) 총지수 MAPE = 0.141%
    하이브리드가 총지수 단계에서도 두 순수 방식보다 낫다(평균오차/바이어스는 -0.008%로
    거의 0). 같은 12개월(2025-08~2026-07) 구간에서 기존 458개 leaf 직접블렌드 방식
    (forecast_august_2026.py의 bottom_up())의 MAPE 0.178%와 비교해도 하이브리드
    (0.150%)가 더 정확했다 — 근거: hybrid_total_backtest_24m.csv.

    다만 **단일 개월 기준으로는 하이브리드가 항상 이기지 않는다**(2026-08 한 달만
    보면 하이브리드 오차 +0.234%로 순수탑다운의 +0.102%보다 오히려 나빴다 —
    apply_hybrid_august2026.py 참조). 24개월 평균으로 신뢰할 방식이지, 특정 한 달의
    결과만으로 우열을 재판단하면 안 된다는 게 이 프로젝트의 반복되는 교훈이다.
    """
    return float(np.average(hybrid_df["최종예측"], weights=hybrid_df["가중치"]))
