"""
국토부 실거래가 원자료(molit_raw/) -> "전월 11일~당월 10일" 기준 전세/월세 변동률 계산 및 CPI 검증

1. 보증금 -> 월세환산: 환산월세(만원/월) = 보증금(만원) * 2.5% * 1.05 / 12
   - 전세: 위 환산월세 그 자체가 조정된 월세금액
   - 월세: 실제 월세금 + (월세보증금의 환산월세)  <- 보증금+월세 혼합구조를 순수 월세환산액으로 통일
2. 개별 거래를 계약년월/계약일로 실제 계약일자를 복원하고, "전월11일~당월10일"을 하나의 관측월로 재구성
   (예: window "2026-08" = 2026-07-11 ~ 2026-08-10)
3. 창(window)별 평균을 두 가지로 계산:
   - 단순평균: 전국 거래 단순평균
   - 지역가중평균: 시도별 평균을 낸 뒤, 표본기간(전체 원자료) 전체의 시도별 거래량 비중을
     고정 가중치로 사용해 합산 (CPI의 "고정가중치 x 변동가격" 방식과 동일한 구조).
     주의: 통계청이 공식적으로 공개한 전세/월세 지역별 가중치가 아니라, 우리가 가진 실거래
     표본의 지역별 거래량 비중을 근사 가중치로 쓴 것 (공식 가중치 미확보 - README 참고)
4. 두 시리즈의 전월비(%)를 CPI 패널의 "전세"/"월세" 실제 전월비(%)와 비교/상관계수 계산

전제: collect_molit_raw_recent.py 로 processed/molit_raw/*.csv 가 미리 확보돼 있어야 한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
PROC_DIR = CODE_DIR / "processed"
RAW_DIR = PROC_DIR / "molit_raw"

CONVERT_RATE = 0.025 * 1.05  # 보증금 -> 연임대료 환산율 (보증금의 2.5%에 임대료상승률 5%를 반영)


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"{RAW_DIR} 에 원자료가 없습니다. 먼저 collect_molit_raw_recent.py 를 실행하세요."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        if df.empty:
            continue
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df


def find_col(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # 부분일치 fallback
    for c in candidates:
        matches = [col for col in df.columns if c in col]
        if matches:
            return matches[0]
    raise KeyError(f"컬럼을 찾지 못함: {candidates} (실제 컬럼: {list(df.columns)})")


def parse_contract_date(df: pd.DataFrame) -> pd.Series:
    ym_col = find_col(df, "계약년월")
    d_col = find_col(df, "계약일")
    ym = df[ym_col].astype(str).str.replace(",", "").str.strip()
    day = pd.to_numeric(df[d_col], errors="coerce").fillna(1).astype(int).clip(1, 28)
    dt = pd.to_datetime(ym + day.astype(str).str.zfill(2), format="%Y%m%d", errors="coerce")
    return dt


def window_label(dt: pd.Series) -> pd.Series:
    """전월11일~당월10일 -> 라벨은 '끝나는 달'(=10일이 속한 달) 기준."""
    day = dt.dt.day
    month = dt.dt.month
    year = dt.dt.year
    target_month = np.where(day >= 11, month + 1, month)
    target_year = year + (target_month > 12).astype(int)
    target_month = np.where(target_month > 12, 1, target_month)
    return pd.Series(
        [f"{y:04d}-{m:02d}" for y, m in zip(target_year, target_month)],
        index=dt.index,
    )


def build_converted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dep_col = find_col(df, "보증금(만원)", "보증금")
    rent_col = find_col(df, "월세금(만원)", "월세금")
    type_col = find_col(df, "전월세구분")
    region_col = find_col(df, "시군구")

    df[dep_col] = df[dep_col].astype(str).str.replace(",", "").astype(float)
    df[rent_col] = df[rent_col].astype(str).str.replace(",", "").astype(float)

    df["계약일자"] = parse_contract_date(df)
    df = df.dropna(subset=["계약일자"])
    df["window"] = window_label(df["계약일자"])
    df["시도"] = df[region_col].astype(str).str.split().str[0]

    converted_deposit_rent = df[dep_col] * CONVERT_RATE / 12
    is_jeonse = df[type_col] == "전세"
    df["환산월세"] = np.where(is_jeonse, converted_deposit_rent, df[rent_col] + converted_deposit_rent)
    df["구분"] = np.where(is_jeonse, "전세", "월세")
    return df[["window", "시도", "구분", "환산월세"]]


def simple_avg_by_window(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["window", "구분"])["환산월세"]
        .mean()
        .reset_index()
        .rename(columns={"환산월세": "단순평균"})
    )


def region_weighted_avg_by_window(df: pd.DataFrame) -> pd.DataFrame:
    # 고정 가중치: 표본 전체 기간의 시도별 거래량 비중 (구분별로 따로)
    region_weight = (
        df.groupby(["구분", "시도"]).size().rename("cnt").reset_index()
    )
    region_weight["w"] = region_weight.groupby("구분")["cnt"].transform(lambda x: x / x.sum())

    region_month_avg = (
        df.groupby(["window", "구분", "시도"])["환산월세"].mean().reset_index()
    )
    merged = region_month_avg.merge(region_weight[["구분", "시도", "w"]], on=["구분", "시도"], how="left")

    out = (
        merged.groupby(["window", "구분"])
        .apply(lambda g: pd.Series({"지역가중평균": (g["환산월세"] * g["w"]).sum() / g["w"].sum()}))
        .reset_index()
    )
    return out


def cpi_actual_mom(item_name: str) -> pd.DataFrame:
    panel = pd.read_csv(PROC_DIR / "cpi_panel.csv")
    s = panel[panel["item_name"] == item_name].sort_values("date").set_index("date")["value"]
    mom = s.pct_change() * 100
    return mom.rename("CPI_실제_전월비%").reset_index()


def main():
    raw = load_raw()
    print(f"원자료 로드: {len(raw)}건 (파일 {len(list(RAW_DIR.glob('*.csv')))}개)")

    converted = build_converted(raw)

    simple = simple_avg_by_window(converted)
    weighted = region_weighted_avg_by_window(converted)
    merged = simple.merge(weighted, on=["window", "구분"], how="outer").sort_values(["구분", "window"])

    merged["단순_전월비%"] = merged.groupby("구분")["단순평균"].pct_change() * 100
    merged["지역가중_전월비%"] = merged.groupby("구분")["지역가중평균"].pct_change() * 100

    results = {}
    for gubun, cpi_item in [("전세", "전세"), ("월세", "월세")]:
        sub = merged[merged["구분"] == gubun].copy()
        cpi = cpi_actual_mom(cpi_item)
        # cpi 'date'는 달력월 - window 라벨과 이름은 다르지만 같은 "YYYY-MM" 형식이라 그대로 병합
        cpi = cpi.rename(columns={"date": "window"})
        comp = sub.merge(cpi, on="window", how="inner")
        comp = comp.dropna(subset=["단순_전월비%"])
        results[gubun] = comp

        corr_simple = comp["단순_전월비%"].corr(comp["CPI_실제_전월비%"])
        corr_weighted = comp["지역가중_전월비%"].corr(comp["CPI_실제_전월비%"])

        print(f"\n{'='*70}\n[{gubun}] 11일~10일 기준 전월비 vs CPI 실제 전월비 (n={len(comp)})")
        print(comp[["window", "단순_전월비%", "지역가중_전월비%", "CPI_실제_전월비%"]].round(3).to_string(index=False))
        print(f"상관계수 - 단순평균 방식: {corr_simple:.3f} / 지역가중 방식: {corr_weighted:.3f}")

        comp.to_csv(PROC_DIR / f"molit_11to10_vs_cpi_{gubun}.csv", index=False, encoding="utf-8-sig")

    return results


if __name__ == "__main__":
    main()
