"""
R-ONE "주택종합"(전체 주택유형 통합, 전국) 전세·월세 가격지수 전월대비 변동률로
CPI 전세·월세 전월대비 변동률을 예측하는 회귀모델.

데이터: 2025-08~2026-07(12개월) - R-ONE 차트(chgRate/tradeChgStatsPage.do)가
표시하는 최대 범위가 12개월로 고정되어 있어(더 이전 시작일을 요청해도 동일하게
잘림을 확인) 이 12개월이 현재 확보 가능한 전부다. 표본이 작아 계수 자체보다는
"어느 정도 관계가 있는지, 어느 방향인지"를 보는 예비모델로 취급해야 한다.

핵심 이슈: R-ONE도 CPI와 마찬가지로 익월 15일 발표(CPI는 익월초 발표)라
R-ONE이 CPI보다 오히려 늦게 나온다 -> 동시점(contemporaneous) 모델은
"사후 설명"용일 뿐 실제 예측(nowcasting)에는 못 쓴다. 대신 "R-ONE(t-1) ->
CPI(t)" 1개월 시차모델은 실제로 미리 알 수 있는 값으로 다음달 CPI를
예측하는 데 쓸 수 있다(단, 표본 11쌍으로 더 작음).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

SCRIPTS = Path(__file__).resolve().parent
MONTHS = ["2025-08","2025-09","2025-10","2025-11","2025-12","2026-01",
          "2026-02","2026-03","2026-04","2026-05","2026-06","2026-07"]

RONE_JEONSE = [0.04,0.10,0.18,0.24,0.28,0.27,0.22,0.28,0.31,0.35,0.38,0.38]
RONE_WOLSE  = [0.10,0.13,0.19,0.23,0.27,0.26,0.24,0.29,0.34,0.35,0.38,0.38]


def cpi_mom(item):
    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv")
    row = panel[panel["품목명"] == item].iloc[0]
    vals = [row["2025-07"]] + [row[m] for m in MONTHS]
    return [(vals[i+1]-vals[i])/vals[i]*100 for i in range(len(MONTHS))]


def fit_report(x, y, label):
    res = stats.linregress(x, y)
    print(f"  [{label}] n={len(x)}  기울기={res.slope:.4f}  절편={res.intercept:.4f}  "
          f"R²={res.rvalue**2:.4f}  p-value={res.pvalue:.4f}")
    return res


def main():
    cpi_jeonse = cpi_mom("전세")
    cpi_wolse = cpi_mom("월세")

    df = pd.DataFrame({
        "년월": MONTHS, "RONE_전세": RONE_JEONSE, "RONE_월세": RONE_WOLSE,
        "CPI_전세": cpi_jeonse, "CPI_월세": cpi_wolse,
    })
    df.to_csv(SCRIPTS / "rone_vs_cpi_monthly.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    print("\n=== 동시점(contemporaneous) 회귀: CPI(t) ~ R-ONE(t) ===")
    print("(주의: R-ONE도 CPI와 비슷하거나 더 늦게 발표되므로 이 모델은 사후설명용)")
    r_jeonse = fit_report(df["RONE_전세"], df["CPI_전세"], "전세")
    r_wolse = fit_report(df["RONE_월세"], df["CPI_월세"], "월세")

    print("\n=== 1개월 시차 회귀: CPI(t) ~ R-ONE(t-1) ===")
    print("(R-ONE 전월치는 CPI 당월 발표 전에 이미 알 수 있는 선행정보 -> 실전 예측에 사용 가능)")
    lag_x_jeonse = RONE_JEONSE[:-1]
    lag_y_jeonse = cpi_jeonse[1:]
    lag_x_wolse = RONE_WOLSE[:-1]
    lag_y_wolse = cpi_wolse[1:]
    rl_jeonse = fit_report(lag_x_jeonse, lag_y_jeonse, "전세(lag1)")
    rl_wolse = fit_report(lag_x_wolse, lag_y_wolse, "월세(lag1)")

    print("\n=== 이 시차모델로 2026-08 CPI 전세/월세 전망 ===")
    print("(2026-07 R-ONE 값을 입력으로 사용 - 2026-08 R-ONE은 9/15 발표라 아직 없음)")
    rone_jul_jeonse = RONE_JEONSE[-1]
    rone_jul_wolse = RONE_WOLSE[-1]
    pred_aug_jeonse_mom = rl_jeonse.slope * rone_jul_jeonse + rl_jeonse.intercept
    pred_aug_wolse_mom = rl_wolse.slope * rone_jul_wolse + rl_wolse.intercept
    print(f"  전세: R-ONE(7월)={rone_jul_jeonse}% -> CPI 전세 8월 전월대비 예측 {pred_aug_jeonse_mom:.4f}%")
    print(f"  월세: R-ONE(7월)={rone_jul_wolse}% -> CPI 월세 8월 전월대비 예측 {pred_aug_wolse_mom:.4f}%")

    panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv")
    jeonse_jul = panel[panel["품목명"] == "전세"].iloc[0]["2026-07"]
    wolse_jul = panel[panel["품목명"] == "월세"].iloc[0]["2026-07"]
    pred_aug_jeonse_idx = jeonse_jul * (1 + pred_aug_jeonse_mom/100)
    pred_aug_wolse_idx = wolse_jul * (1 + pred_aug_wolse_mom/100)
    print(f"\n  전세지수 8월 예측(R-ONE시차모델): {pred_aug_jeonse_idx:.3f} (기존 ETS예측 106.33과 비교)")
    print(f"  월세지수 8월 예측(R-ONE시차모델): {pred_aug_wolse_idx:.3f} (기존 ETS예측 105.95와 비교)")


if __name__ == "__main__":
    main()
