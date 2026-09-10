"""
2026년 9월 CPI 예측 — forecast_august_2026.py와 동일한 바텀업/탑다운 파이프라인에
게이트 편향보정(apply_bias_correction_august.py)과 하이브리드 결합
(hybrid_category_model.py)까지 한 스크립트에서 순서대로 실행한다.

8월과의 핵심 차이:
1) 학습 데이터가 2026-08까지 확장됨(update_panels_with_august_actual.py로 실제
   8월 발표치를 패널에 반영 완료).
2) Tier D 오피넷 회귀변수는 9월 MTD 데이터가 4일치(09/01~09/06)뿐이라 8월(24일치)보다
   훨씬 얇다 - raw_opinet_gasoline_diesel_kerosene.tsv/raw_opinet_auto_lpg.tsv에 추가.
   취사용LPG·부탄가스는 9월 MTD 자체가 없어(월간 확정치라 월초에나 나옴) 회귀변수
   없이 드리프트 추정으로 자동 폴백됨.
3) 카테고리/총지수는 하이브리드(04·06·09만 탑다운, 나머지 바텀업)를 공식 예측으로 채택.
4) 전세·월세는 이번엔 R-ONE 주간 데이터 갱신 없이 자체 ETS 그대로 사용(9월 R-ONE
   주간치가 아직 1주 안팎이라 이번 회차에는 보류 - 데이터 부족 섹션에 기록).
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from hybrid_category_model import bottomup_to_category, combine_hybrid, hybrid_total

warnings.filterwarnings("ignore")

CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"
TARGET = "2026-09"

FIXED_SEASON_NAMES = {
    "복숭아", "포도", "감", "귤", "오렌지", "참외", "수박", "딸기", "체리", "열무", "굴",
}
ANNUAL_STEP_NAMES = {
    "외래진료비", "한방진료비", "약국조제료", "치과진료비", "입원진료비",
    "전문대학납입금", "국공립대학교납입금", "사립대학교납입금",
    "국공립대학원납입금", "사립대학원납입금",
    # 2026-09-10 검증: 장기요양 수가가 매년 11~12월경 인상 반영되는 패턴이 확인됨
    # (2023-12/2024-11/2025-12 모두 스파이크). 24개월 백테스트 MAPE 0.57%->0.38%로
    # 개선(34%↓) - 검증 후 추가.
    "요양시설이용료",
}
# 2026-09-10 검증(전기료 발견 후 Tier A 이외 전체를 훑어봄): 유치원납입금도 매년
# 7월~2월에 반복적으로 큰 하락(-4~19%)이 나타나 계절패턴처럼 보였으나, 실제
# 계절ETS로 백테스트하니 MAPE가 2.72%->2.79%로 오히려 소폭 악화됨(패턴이 매년
# 시기/크기가 불규칙해 12개월 주기 가정이 안 맞음) - 가설은 기각, 비계절ETS 유지.
# 도시가스·시내버스료·택시료·도시철도료·하수도료·지역난방비 등은 "매년 같은 달"이
# 아니라 비정기적 요금고시(뉴스 기반 수동추적이 필요한 진짜 Tier C 케이스)라 계절
# ETS로 고쳐지는 문제가 아님 - 대상에서 제외.
ANNUAL_SEASONAL_TIER_D = {"전기료"}
# 2026-09-10 검증: 6년치(2020-09~2026-08) 변동월 클러스터링으로 "동결이 잦으면서
# 변동이 특정 달에 몰리는" 품목을 스캔한 결과, 의류 5종이 변동의 절반 이상이
# 11월(겨울 신상 출시기)에 몰려있었고 24개월 백테스트에서 실제로 개선 확인:
#   여자외의 0.341%->0.323%, 점퍼 0.373%->0.274%, 여자하의 0.432%->0.348%,
#   남자상의 0.243%->0.183%, 남자하의 0.246%->0.229%, 쓰레기봉투료(1월 집중) 0.102%->0.082%
# 반면 같은 스캔에서 나온 티셔츠·스웨터·등산복·보청기·이러닝이용료·대입전형료는
# 백테스트에서 오히려 악화되어(예: 스웨터 0.121%->0.281%) 제외 - 시각적 패턴만으론
# 판단하지 않고 전부 개별 검증했다.
ANNUAL_SEASONAL_MISC = {
    "여자외의", "점퍼", "여자하의", "남자상의", "남자하의", "쓰레기봉투료",
    # 2026-09-10 재검증(전체 26년치 2000-01~2026-08로 재스캔, 6년 스캔에서는 임계값을
    # 못 넘었던 항목까지 확인): 유아동복(5월 집중 45.6%) 0.264%->0.246%, 청바지(11월
    # 집중 50%) 0.340%->0.255% 개선 확인 후 추가. 같은 스캔에서 나온 여자상의(5월
    # 집중 50%)는 0.171%->0.280%로 악화되어 제외 - 월 집중도가 비슷해도 결과가
    # 갈리므로 매번 개별 백테스트가 필수.
    "유아동복", "청바지",
}
# 참고(적용 안 함): 휴대전화기(가중치10.4, 31/35개월 동결)도 계단식 패턴처럼 보여
# 2026-09-10 재검증했으나 여전히 계절ETS가 크게 악화됨(비계절 0.44% -> 계절 3.08%,
# 수개월 전 최초 시도 때의 0.468%->3.069%와 거의 동일 - 재현성 확인). 비계절ETS 유지.
OPINET_REGRESSOR = {
    "휘발유": ("raw_opinet_gasoline_diesel_kerosene.tsv", "보통휘발유"),
    "경유": ("raw_opinet_gasoline_diesel_kerosene.tsv", "자동차용경유"),
    "등유": ("raw_opinet_gasoline_diesel_kerosene.tsv", "실내등유"),
    "자동차용LPG": ("raw_opinet_auto_lpg.tsv", "자동차부탄(원/L)"),
    "취사용LPG": ("raw_opinet_household_lpg.tsv", "일반프로판(원/kg)"),
}

TRAILING_MONTHS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07"]
SHRINKAGE = 0.6
CONSISTENCY_MIN = 0.8
STD_MAX_PCT = 2.0


def load_opinet_series(fname, col):
    df = pd.read_csv(SCRIPTS / fname, sep="\t", dtype=str)
    df = df.rename(columns={df.columns[0]: "기간"})
    if "년" in str(df["기간"].iloc[0]):
        extracted = df["기간"].str.extract(r"(\d{4})년(\d{2})월")
        df["기간"] = extracted[0] + "-" + extracted[1]
    else:
        df["기간"] = df["기간"].str.extract(r"(\d{4}-\d{2})")[0]
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    return df.set_index("기간")[col]


def ets_forecast(hist: pd.Series, is_seasonal: bool) -> float:
    s = hist.dropna()
    if len(s) < 24:
        if len(s) >= 2:
            drift = s.diff().dropna().tail(12).mean()
            return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))
        return float(s.iloc[-1]) if len(s) else np.nan
    try:
        if is_seasonal and len(s) >= 36:
            model = ExponentialSmoothing(s.values, trend="add", damped_trend=True,
                                          seasonal="add", seasonal_periods=12,
                                          initialization_method="estimated")
        else:
            model = ExponentialSmoothing(s.values, trend="add", damped_trend=True,
                                          seasonal=None, initialization_method="estimated")
        fit = model.fit(optimized=True)
        return float(fit.forecast(1)[0])
    except Exception:
        drift = s.diff().dropna().tail(12).mean()
        return float(s.iloc[-1] + (drift if pd.notna(drift) else 0))


def step_weighted_forecast(hist: pd.Series, target_month_num: int, weights=(5, 4, 3, 2, 1)) -> float:
    """예측 대상월과 같은 '달'로의 과거 전월대비 변동만 모아 최근 N회를 가중평균
    (최신일수록 가중치 큼: 5,4,3,2,1)해 그 달의 기대 변동률로 쓴다. 다른 달로의
    전이는 원래 변동이 거의 없으니 자연히 0%에 가깝게 나온다 - "그 달에만 계단식
    으로 움직이고 나머진 안 움직인다"는 패턴을 정면으로 모델링한 나이브 계절 기법.
    select_item_model.py의 24개월 백테스트로 검증된 품목에만 적용한다."""
    s = hist.dropna()
    pct = s.pct_change().dropna() * 100
    same_month = pct[pd.Index(pct.index).map(lambda x: int(x[5:7])) == target_month_num]
    if len(same_month) == 0:
        return float(s.iloc[-1])
    recent = same_month.tail(len(weights))
    w = np.array(weights[-len(recent):], dtype=float)
    avg_pct = float(np.average(recent.values, weights=w))
    return float(s.iloc[-1] * (1 + avg_pct / 100))


def regression_forecast(cpi_hist: pd.Series, ext_hist: pd.Series, ext_target_value: float) -> float:
    common = sorted(set(cpi_hist.dropna().index) & set(ext_hist.dropna().index))
    if len(common) < 24 or pd.isna(ext_target_value):
        drift = cpi_hist.dropna().diff().dropna().tail(12).mean()
        return float(cpi_hist.dropna().iloc[-1] + (drift if pd.notna(drift) else 0))
    y = cpi_hist[common].astype(float).values
    x = ext_hist[common].astype(float).values
    b, a = np.polyfit(x, y, 1)
    return float(a + b * ext_target_value)


def load_item_model_selection() -> dict:
    """select_item_model.py의 24개월 백테스트 결과(item_model_selection.csv)에서
    품목별 채택방식('계단평균'/'완전동결'/'계절ETS' - '비계절ETS'는 현행유지라 제외)을
    읽는다. 파일이 없으면 빈 dict(전부 기존 로직 유지)."""
    path = SCRIPTS / "item_model_selection.csv"
    if not path.exists():
        return {}
    sel = pd.read_csv(path, encoding="utf-8-sig")
    sel = sel[sel["채택방식"] != "비계절ETS"]
    return dict(zip(sel["품목명"], sel["채택방식"]))


def bottom_up():
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv")
    month_cols_sorted = sorted([c for c in panel.columns if c[:2] in ("19", "20")])
    train_cols = [c for c in month_cols_sorted if c < TARGET]
    target_month_num = int(TARGET[5:7])

    opinet_cache = {name: load_opinet_series(f, c) for name, (f, c) in OPINET_REGRESSOR.items()}
    item_selection = load_item_model_selection()

    rows = []
    for _, row in panel.iterrows():
        item, tier, weight = row["품목명"], row["Tier"], row["가중치"]
        full_series = pd.Series(row[month_cols_sorted].astype(float).values,
                                 index=month_cols_sorted).interpolate(limit_area="inside")
        cpi_hist = full_series[train_cols]

        if item in OPINET_REGRESSOR:
            ext_series = opinet_cache[item]
            ext_hist = ext_series[ext_series.index.isin(train_cols)]
            ext_target = ext_series.get(TARGET, np.nan)
            pred = regression_forecast(cpi_hist, ext_hist, ext_target)
            model_used = "회귀(오피넷 9월실측)" if pd.notna(ext_target) else "회귀변수없음->드리프트"
        elif item_selection.get(item) == "계단평균":
            pred = step_weighted_forecast(cpi_hist, target_month_num)
            model_used = "계단평균(검증됨)"
        elif item_selection.get(item) == "완전동결":
            pred = float(cpi_hist.dropna().iloc[-1])
            model_used = "완전동결(검증됨)"
        else:
            if tier == "B":
                is_seasonal, restrict = True, item in FIXED_SEASON_NAMES
            elif tier == "C" and item in ANNUAL_STEP_NAMES:
                is_seasonal, restrict = True, False
            elif item in ANNUAL_SEASONAL_TIER_D:
                is_seasonal, restrict = True, False
            elif item in ANNUAL_SEASONAL_MISC:
                is_seasonal, restrict = True, False
            elif item_selection.get(item) == "계절ETS":
                is_seasonal, restrict = True, False
            else:
                is_seasonal, restrict = False, False
            hist = cpi_hist[cpi_hist.index >= "2017-01"] if restrict else cpi_hist
            pred = ets_forecast(hist, is_seasonal)
            model_used = "계절ETS" if is_seasonal else "비계절ETS"

        rows.append({"품목코드": row["품목코드"], "품목명": item, "Tier": tier,
                      "가중치": weight, "모델": model_used, f"pred_{TARGET}": pred})

    out = pd.DataFrame(rows)
    out.to_csv(SCRIPTS / "september2026_bottomup_items.csv", index=False, encoding="utf-8-sig")

    w = out["가중치"].values
    vals = out[f"pred_{TARGET}"].astype(float).values
    arith = np.average(vals, weights=w)
    geom = np.exp(np.average(np.log(vals), weights=w))
    blended = 0.5 * arith + 0.5 * geom
    return out, blended


def apply_bias_correction(bottomup_items: pd.DataFrame) -> pd.DataFrame:
    """항목별_편향보정_결과.md에서 검증된 게이트 기반 트레일링 편향보정 (그대로 재사용)."""
    backtest = pd.read_csv(SCRIPTS / "all_tiers_forecast_vs_actual.csv")
    df = bottomup_items.copy()

    corrections = {}
    for _, row in backtest.iterrows():
        item = row["품목명"]
        lvl = [row[f"pred_{m}"] - row[f"actual_{m}"] for m in TRAILING_MONTHS]
        pct = [(row[f"pred_{m}"] - row[f"actual_{m}"]) / row[f"actual_{m}"] * 100 for m in TRAILING_MONTHS]
        mean_pct = np.mean(pct)
        consistency = np.mean([np.sign(p) == np.sign(mean_pct) for p in pct])
        std_pct = np.std(pct)
        if consistency >= CONSISTENCY_MIN and std_pct <= STD_MAX_PCT and abs(mean_pct) >= 0.05:
            corrections[item] = np.mean(lvl)
        else:
            corrections[item] = 0.0

    pred_col = f"pred_{TARGET}"
    df["보정치(레벨)"] = df["품목명"].map(corrections).fillna(0.0)
    df[f"보정후_pred_{TARGET}"] = df[pred_col] - SHRINKAGE * df["보정치(레벨)"]
    df.to_csv(SCRIPTS / "september2026_bottomup_items_corrected.csv", index=False, encoding="utf-8-sig")

    n_corrected = (df["보정치(레벨)"] != 0).sum()
    print(f"편향보정 적용 품목수: {n_corrected} / {len(df)} (트레일링 {TRAILING_MONTHS[0]}~{TRAILING_MONTHS[-1]}, "
          f"7월까지의 백테스트 기준 - 8월 신규 백테스트 포인트는 미반영, 다음 개선과제)")
    return df


def top_down():
    official = pd.read_csv(SCRIPTS / "cpi_official_monthly_wide.csv").drop_duplicates(subset="품목", keep="first")
    month_cols_sorted = sorted([c for c in official.columns if c[:2] in ("19", "20")])
    train_cols = [c for c in month_cols_sorted if c < TARGET]

    total_row = official[official["품목"] == "0 총지수"].iloc[0]
    major_rows = official[official["품목"].str.match(r"^\d{2} ")]
    sub_rows = official[official["품목"].str.match(r"^\d{2}\.\d ")]

    results = []
    for label, r in [("총지수", total_row)] + list(zip(major_rows["품목"], major_rows.to_dict("records"))) + \
                     list(zip(sub_rows["품목"], sub_rows.to_dict("records"))):
        series = pd.Series({c: r[c] for c in month_cols_sorted}).astype(float)
        hist = series[train_cols].dropna()
        pred = ets_forecast(hist, is_seasonal=True)
        last_actual = hist.iloc[-1] if len(hist) else np.nan
        results.append({
            "분류": label, "2026-08(실제)": last_actual, f"{TARGET}(예측)": pred,
            "전월대비%": (pred - last_actual) / last_actual * 100 if last_actual else np.nan,
        })

    out = pd.DataFrame(results)
    out.to_csv(SCRIPTS / "september2026_topdown_categories.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    print(f"=== 9월 CPI 예측 (TARGET={TARGET}) ===\n")

    print("[1/4] 바텀업(458개 품목)")
    items, blended_raw = bottom_up()
    print(f"  보정 전 바텀업 총지수: {blended_raw:.3f}")

    print("\n[2/4] 게이트 편향보정")
    items_corrected = apply_bias_correction(items)
    w = items_corrected["가중치"].values
    vals = items_corrected[f"보정후_pred_{TARGET}"].values
    arith = np.average(vals, weights=w)
    geom = np.exp(np.average(np.log(vals), weights=w))
    blended_corrected = 0.5 * arith + 0.5 * geom
    print(f"  보정 후 바텀업 총지수: {blended_corrected:.3f}")

    print("\n[3/4] 탑다운(대분류12 + 소분류38 + 총지수)")
    cat_df = top_down()
    major = cat_df[cat_df["분류"].str.match(r"^\d\d ")]
    print(major[["분류", "2026-08(실제)", f"{TARGET}(예측)", "전월대비%"]].round(3).to_string(index=False))

    print("\n[4/4] 하이브리드 결합 (04·06·09만 탑다운, 나머지 바텀업)")
    bu_cat = bottomup_to_category(items_corrected, pred_col=f"보정후_pred_{TARGET}")
    td_major = major.rename(columns={f"{TARGET}(예측)": "탑다운예측"})[["분류", "탑다운예측"]]
    hybrid = combine_hybrid(bu_cat, td_major)
    hybrid = hybrid.sort_values("분류")
    hybrid.to_csv(SCRIPTS / "september2026_hybrid_category.csv", index=False, encoding="utf-8-sig")
    print(hybrid[["분류", "바텀업예측", "탑다운예측", "채택방식", "최종예측"]].round(3).to_string(index=False))

    total = hybrid_total(hybrid)
    total_row = cat_df[cat_df["분류"] == "총지수"].iloc[0]
    print(f"\n[총지수 예측 비교]")
    print(f"  바텀업(458개 직접블렌드, 편향보정 후): {blended_corrected:.3f}")
    print(f"  탑다운(총지수 자체 히스토리 ETS):        {total_row[f'{TARGET}(예측)']:.3f}")
    print(f"  하이브리드(카테고리별 결합 후 재구성):     {total:.3f}  <- 공식 채택값")
    aug_actual = total_row["2026-08(실제)"]
    print(f"  (8월 실제: {aug_actual:.2f}, 전월대비 {(total-aug_actual)/aug_actual*100:+.3f}%)")


if __name__ == "__main__":
    main()
