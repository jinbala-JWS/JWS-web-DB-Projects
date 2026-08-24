"""
R-ONE 주간 아파트 전세가격 3주치(8월)로 8월 아파트 전세 변동률을 추정하고,
연립다세대·단독주택은 과거 12개월간 아파트 대비 관계(회귀)로 추정한 뒤,
Jevons 가중치(아파트62.69%/연립다세대25.23%/단독주택12.08%,
R-ONE_주택종합지수_가중치_분석.md에서 역산)로 "주택종합" 8월 전세 변동률을 합성.
이 값을 동시점 회귀모델(CPI(t)~R-ONE(t), R-ONE_CPI_전월세_회귀모델.md)에 넣어
CPI 전세 8월 상승률을 예측한다.
"""
import numpy as np
from scipy import stats

# 1) 8월 주간 아파트 전세 변동률(전주대비 %) - R-ONE 주간아파트가격동향, 2026년 8월 1~3주차
WEEKLY_AUG_APT_JEONSE = [0.11, 0.09, 0.10]  # 08.03, 08.10, 08.17

# 2) 월별(2025-08~2026-07) 아파트/연립다세대/단독주택 전세 변동률(%) - R-ONE 차트 API 실측
MONTHS = ["2025-08","2025-09","2025-10","2025-11","2025-12","2026-01",
          "2026-02","2026-03","2026-04","2026-05","2026-06","2026-07"]
APT = [0.06,0.14,0.24,0.33,0.38,0.37,0.30,0.38,0.41,0.45,0.49,0.49]
ROW = [0.03,0.08,0.15,0.11,0.14,0.10,0.09,0.12,0.18,0.21,0.28,0.27]      # 연립다세대
DET = [-0.01,0.00,0.00,0.01,0.01,0.02,0.02,0.03,0.03,0.04,0.03,0.04]     # 단독주택

# 3) Jevons 가중치(역산, R-ONE_주택종합지수_가중치_분석.md)
W_APT, W_ROW, W_DET = 0.6269, 0.2523, 0.1208

# 4) CPI(t) ~ R-ONE 종합(t) 동시점 회귀 (R-ONE_CPI_전월세_회귀모델.md)
CPI_SLOPE, CPI_INTERCEPT = 0.2726, 0.0196


def compound(rates_pct):
    """전주대비(%) 리스트를 복리로 합성한 누적변동률(%)"""
    factor = 1.0
    for r in rates_pct:
        factor *= (1 + r / 100)
    return (factor - 1) * 100


def main():
    # --- 8월 아파트 전세: 3주 복리합성 ---
    apt_aug = compound(WEEKLY_AUG_APT_JEONSE)
    print(f"8월 아파트 전세(주간 3주 복리합성): {apt_aug:.4f}%")

    # 검증: 7월 4주 복리합성이 실제 7월 월간치(0.49%)와 얼마나 가까운지 체크
    # (7월 주간: 07.06=0.12, 07.13=0.11, 07.20=0.11, 07.27=0.10)
    jul_check = compound([0.12, 0.11, 0.11, 0.10])
    calib_ratio = jul_check / 0.49
    print(f"[검산] 7월 주간 4주 복리합성={jul_check:.4f}% vs 실제 7월 월간치=0.49% "
          f"(비율 {calib_ratio*100:.1f}%)")
    apt_aug_calib = apt_aug / calib_ratio
    print(f"[보정] 이 비율로 8월치 보정: {apt_aug:.4f}% -> {apt_aug_calib:.4f}%"
          f" (7월과 동일하게 주 수가 하나 적어 과소추정됐을 가능성 보정, 참고용)")

    # --- 연립다세대/단독주택: 아파트 대비 과거 12개월 관계로 회귀추정 ---
    row_fit = stats.linregress(APT, ROW)
    det_fit = stats.linregress(APT, DET)
    print(f"\n연립다세대 ~ 아파트 회귀: 기울기={row_fit.slope:.4f} 절편={row_fit.intercept:.4f} R²={row_fit.rvalue**2:.4f}")
    print(f"단독주택   ~ 아파트 회귀: 기울기={det_fit.slope:.4f} 절편={det_fit.intercept:.4f} R²={det_fit.rvalue**2:.4f}")

    jeonse_jul_idx = 106.17  # CPI 전세지수 2026-07 실제값

    def pipeline(apt_val, label):
        row_v = row_fit.slope * apt_val + row_fit.intercept
        det_v = det_fit.slope * apt_val + det_fit.intercept
        comp_v = (
            (1 + apt_val/100) ** W_APT * (1 + row_v/100) ** W_ROW * (1 + det_v/100) ** W_DET - 1
        ) * 100
        cpi_mom = CPI_SLOPE * comp_v + CPI_INTERCEPT
        cpi_idx = jeonse_jul_idx * (1 + cpi_mom/100)
        print(f"\n--- {label} (아파트 {apt_val:.4f}%) ---")
        print(f"  연립다세대(추정) {row_v:.4f}% / 단독주택(추정) {det_v:.4f}%")
        print(f"  R-ONE 주택종합(합성) {comp_v:.4f}%")
        print(f"  => CPI 전세 8월 전월대비 예측 {cpi_mom:.4f}%  => 지수 예측 {cpi_idx:.3f}")
        return cpi_idx

    idx_raw = pipeline(apt_aug, "원값(3주 복리합성 그대로)")
    idx_calib = pipeline(apt_aug_calib, "보정값(7월 검산비율로 상향보정)")

    print(f"\n[종합] 8월 CPI 전세지수 예측 범위: {min(idx_raw,idx_calib):.3f} ~ {max(idx_raw,idx_calib):.3f}")
    print("   (참고: 기존 ETS예측 106.33, R-ONE월간시차모델예측 106.317)")


if __name__ == "__main__":
    main()
