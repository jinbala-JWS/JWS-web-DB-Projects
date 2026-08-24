"""
전세와 같은 방식으로 8월 CPI 월세 상승률을 추산.

차이점: R-ONE 주간통계는 매매·전세만 조사하고 월세는 없음(주간 데이터의
BLD_GB가 01/02만 존재함을 실측으로 확인). 따라서 "8월 아파트 전세"를 주간
3주치로 추정한 뒤(predict_august_via_weekly_apt.py와 동일), 과거 12개월간
"아파트 월세 ~ 아파트 전세" 관계를 회귀로 학습해 8월 아파트 월세로 변환한다.
그 다음은 전세와 동일한 절차(연립다세대·단독주택 추정 -> Jevons 합성 -> CPI 회귀).
"""
import numpy as np
from scipy import stats

WEEKLY_AUG_APT_JEONSE = [0.11, 0.09, 0.10]  # 08.03, 08.10, 08.17 (전세, 월세는 주간데이터 없음)

MONTHS = ["2025-08","2025-09","2025-10","2025-11","2025-12","2026-01",
          "2026-02","2026-03","2026-04","2026-05","2026-06","2026-07"]
APT_JEONSE = [0.06,0.14,0.24,0.33,0.38,0.37,0.30,0.38,0.41,0.45,0.49,0.49]
APT_WOLSE  = [0.13,0.16,0.24,0.29,0.34,0.33,0.30,0.36,0.42,0.43,0.47,0.47]
ROW_WOLSE  = [0.08,0.13,0.16,0.18,0.20,0.19,0.16,0.19,0.25,0.27,0.32,0.31]  # 연립다세대
DET_WOLSE  = [0.03,0.03,0.03,0.04,0.06,0.05,0.05,0.05,0.06,0.07,0.06,0.06]  # 단독주택

W_APT, W_ROW, W_DET = 0.6269, 0.2523, 0.1208
CPI_SLOPE, CPI_INTERCEPT = 0.2454, 0.0352  # CPI 월세(t) ~ R-ONE 월세종합(t) 동시점회귀

WOLSE_JUL_IDX = 105.82  # CPI 월세지수 2026-07 실제값


def compound(rates_pct):
    factor = 1.0
    for r in rates_pct:
        factor *= (1 + r / 100)
    return (factor - 1) * 100


def main():
    apt_jeonse_aug = compound(WEEKLY_AUG_APT_JEONSE)
    jul_check = compound([0.12, 0.11, 0.11, 0.10])
    calib_ratio = jul_check / 0.49
    apt_jeonse_aug_calib = apt_jeonse_aug / calib_ratio
    print(f"8월 아파트 전세(주간 3주 복리합성): 원값 {apt_jeonse_aug:.4f}% / 보정값 {apt_jeonse_aug_calib:.4f}%")

    # 아파트 월세 ~ 아파트 전세 (같은 유형 내 전세->월세 변환)
    aw_fit = stats.linregress(APT_JEONSE, APT_WOLSE)
    print(f"\n아파트 월세 ~ 아파트 전세 회귀: 기울기={aw_fit.slope:.4f} 절편={aw_fit.intercept:.4f} R²={aw_fit.rvalue**2:.4f}")

    # 연립다세대/단독주택 월세 ~ 아파트 월세
    row_fit = stats.linregress(APT_WOLSE, ROW_WOLSE)
    det_fit = stats.linregress(APT_WOLSE, DET_WOLSE)
    print(f"연립다세대 월세 ~ 아파트 월세 회귀: 기울기={row_fit.slope:.4f} 절편={row_fit.intercept:.4f} R²={row_fit.rvalue**2:.4f}")
    print(f"단독주택 월세   ~ 아파트 월세 회귀: 기울기={det_fit.slope:.4f} 절편={det_fit.intercept:.4f} R²={det_fit.rvalue**2:.4f}")

    def pipeline(apt_jeonse_val, label):
        apt_wolse_v = aw_fit.slope * apt_jeonse_val + aw_fit.intercept
        row_v = row_fit.slope * apt_wolse_v + row_fit.intercept
        det_v = det_fit.slope * apt_wolse_v + det_fit.intercept
        comp_v = (
            (1 + apt_wolse_v/100) ** W_APT * (1 + row_v/100) ** W_ROW * (1 + det_v/100) ** W_DET - 1
        ) * 100
        cpi_mom = CPI_SLOPE * comp_v + CPI_INTERCEPT
        cpi_idx = WOLSE_JUL_IDX * (1 + cpi_mom/100)
        print(f"\n--- {label} ---")
        print(f"  아파트 월세(변환추정) {apt_wolse_v:.4f}%")
        print(f"  연립다세대 월세(추정) {row_v:.4f}% / 단독주택 월세(추정) {det_v:.4f}%")
        print(f"  R-ONE 주택종합 월세(합성) {comp_v:.4f}%")
        print(f"  => CPI 월세 8월 전월대비 예측 {cpi_mom:.4f}%  => 지수 예측 {cpi_idx:.3f}")
        return cpi_idx

    idx_raw = pipeline(apt_jeonse_aug, "원값")
    idx_calib = pipeline(apt_jeonse_aug_calib, "보정값")

    print(f"\n[종합] 8월 CPI 월세지수 예측 범위: {min(idx_raw,idx_calib):.3f} ~ {max(idx_raw,idx_calib):.3f}")
    print("   (참고: 기존 ETS예측 105.95, R-ONE월간시차모델예측 105.959)")


if __name__ == "__main__":
    main()
