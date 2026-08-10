"""
CPI_Tier_분류.csv 에 품목별 "자동화여부"/"데이터출처" 컬럼을 추가/갱신한다.

models.py, collect_market_prices.py에 실제로 반영된 매핑을 그대로 가져다 쓰므로
코드와 분류표가 따로 놀지 않는다(둘 중 하나만 고치면 표가 어긋나는 문제 방지).
"""

from pathlib import Path

import pandas as pd

import collect_market_prices as cmp
import models

BASE_DIR = Path(__file__).resolve().parent.parent
TIER_CSV = BASE_DIR / "CPI" / "CPI_Tier_분류.csv"


def classify(row) -> tuple[str, str]:
    tier = row["Tier"]
    name = row["품목명"]

    if tier == "A":
        if name in cmp.EKAPE_ITEMS:
            return "자동화 완료", "축산유통정보(다봄, requests 자동수집)"
        if name in cmp.KAMIS_ITEMS:
            return "미자동화(수동조회 필요)", "KAMIS 농산물유통정보 (자동수집 미해결 - 서버 500 오류)"
        return "자동화(외부데이터 없음)", "CPI 자체시계열 (백테스트 기반 모델 자동선택 + 상위분류 축소)"

    if tier == "B":
        return "자동화(외부데이터 없음)", "CPI 자체시계열 (최근5개년 계절중앙값 / 고정계절품목은 통계청 공식 대체로직)"

    if tier == "C":
        return "수동(기본값=보합)", "수동입력 대기 (code/manual_overrides.json, 뉴스·공지 확인 필요)"

    if tier == "D":
        if name in models.TIER_D_EXTERNAL_MAP:
            col = models.TIER_D_EXTERNAL_MAP[name]
            return "자동화 완료", f"오피넷 유가데이터 ({col}, requests 자동수집)"
        if name in models.TIER_D_REGULATED:
            return "수동(기본값=보합)", "수동입력 대기 (전기/가스/난방 요금개정이력, code/manual_overrides.json)"
        return "자동화(외부데이터 없음)", "CPI 자체시계열 (Tier A 방식 폴백, 오피넷에 취급 품목 없음)"

    return "미분류", ""


def main():
    df = pd.read_csv(TIER_CSV, encoding="utf-8-sig")
    results = df.apply(classify, axis=1, result_type="expand")
    df["자동화여부"] = results[0]
    df["데이터출처"] = results[1]
    df.to_csv(TIER_CSV, index=False, encoding="utf-8-sig")

    print(f"업데이트 완료: {TIER_CSV}")
    print("\n[자동화여부 분포]")
    print(df["자동화여부"].value_counts().to_string())
    print("\n[Tier x 자동화여부 교차표]")
    print(pd.crosstab(df["Tier"], df["자동화여부"]).to_string())


if __name__ == "__main__":
    main()
