"""
오피넷에서 수집한 원자재(휘발유/경유/등유/LPG) 가격과
통계청 공식 CPI 세부지수(official_tierCD_monthly.csv)를 비교한다.

비교 방법: 두 시계열 모두 2020년을 100으로 재기준화(rebase)한 뒤
  1) 레벨 상관계수(correlation)
  2) 전월대비 변화율(MoM %) 상관계수 - 방향성/타이밍이 맞는지가 핵심
  3) 최근 12개월 평균 절대 오차(레벨 기준)
을 계산해 "일치 여부"를 정량적으로 판단한다.
"""
import pandas as pd
from pathlib import Path

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"


def load_official():
    df = pd.read_csv(SCRIPTS / "official_tierCD_monthly.csv")
    df = df.set_index("품목명")
    month_cols = [c for c in df.columns if c[:2] in ("19", "20")]
    return df[month_cols]


def rebase_2020(series: pd.Series) -> pd.Series:
    base = series.loc[[c for c in series.index if c.startswith("2020")]].mean()
    return series / base * 100


def load_tsv(name, cols):
    df = pd.read_csv(SCRIPTS / name, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "기간"})
    df["기간"] = df["기간"].astype(str)
    df = df.set_index("기간")
    for c in cols:
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",", "", regex=False).str.strip().replace({"-": None, "": None, "nan": None}),
            errors="coerce",
        )
    return df[cols]


def compare(official_name, official_row, proxy_series, label):
    off = official_row.copy()
    off.index = [c for c in off.index]
    common = sorted(set(off.index) & set(proxy_series.index))
    common = [c for c in common if pd.notna(off[c]) and pd.notna(proxy_series[c])]
    if len(common) < 12:
        print(f"[{label}] 공통 구간 부족 ({len(common)}개월) - 스킵")
        return None

    off_c = off[common].astype(float)
    proxy_c = proxy_series[common].astype(float)

    off_idx = rebase_2020(off_c)
    proxy_idx = rebase_2020(proxy_c)

    level_corr = off_idx.corr(proxy_idx)

    off_mom = off_c.pct_change().dropna()
    proxy_mom = proxy_c.pct_change().dropna()
    mom_common = sorted(set(off_mom.index) & set(proxy_mom.index))
    mom_corr = off_mom[mom_common].corr(proxy_mom[mom_common])

    recent = common[-12:]
    mae_recent = (off_idx[recent] - proxy_idx[recent]).abs().mean()

    print(f"\n[{official_name}] vs [{label}]  (공통 {len(common)}개월: {common[0]} ~ {common[-1]})")
    print(f"  레벨(2020=100 재기준) 상관계수     : {level_corr:.4f}")
    print(f"  전월대비 변화율(MoM) 상관계수      : {mom_corr:.4f}")
    print(f"  최근 12개월 평균 절대오차(레벨기준) : {mae_recent:.2f}")
    return {
        "official": official_name,
        "proxy": label,
        "n_months": len(common),
        "level_corr": level_corr,
        "mom_corr": mom_corr,
        "mae_recent12": mae_recent,
    }


def main():
    official = load_official()

    gasoline = load_tsv(
        "raw_opinet_gasoline_diesel_kerosene.tsv",
        ["보통휘발유", "자동차용경유", "실내등유"],
    )
    gasoline.index = [
        f"{int(i[:4])}-{i[5:7]}" for i in gasoline.index
    ]  # '2000년01월' -> '2000-01'

    auto_lpg = load_tsv("raw_opinet_auto_lpg.tsv", ["자동차부탄(원/L)"])

    hh_lpg = load_tsv("raw_opinet_household_lpg.tsv", ["일반프로판(원/kg)", "일반부탄(원/kg)"])

    results = []
    results.append(compare("휘발유", official.loc["휘발유"], gasoline["보통휘발유"], "오피넷 보통휘발유"))
    results.append(compare("경유", official.loc["경유"], gasoline["자동차용경유"], "오피넷 자동차용경유"))
    results.append(compare("등유", official.loc["등유"], gasoline["실내등유"], "오피넷 실내등유"))
    results.append(compare("자동차용LPG", official.loc["자동차용LPG"], auto_lpg["자동차부탄(원/L)"], "오피넷 자동차부탄"))
    results.append(compare("취사용LPG", official.loc["취사용LPG"], hh_lpg["일반프로판(원/kg)"], "오피넷 일반프로판"))
    results.append(compare("부탄가스", official.loc["부탄가스"], hh_lpg["일반부탄(원/kg)"], "오피넷 일반부탄"))

    results = [r for r in results if r]
    out = pd.DataFrame(results)
    out.to_csv(SCRIPTS / "comparison_opinet_vs_official.csv", index=False, encoding="utf-8-sig")
    print("\n저장:", SCRIPTS / "comparison_opinet_vs_official.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
