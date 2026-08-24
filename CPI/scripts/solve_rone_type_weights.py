"""
R-ONE 전국주택가격동향조사 "주택종합" 전세/월세 전월대비 변동률이
아파트/연립다세대/단독주택 3개 유형의 변동률을 어떤 가중치로 결합해 나온 것인지 역산.

공식 방법론(2026.7 전국주택가격동향조사 보고서 32페이지):
"주택재고량을 가중치로 한 제본스지수 산식(Jevons index formula)으로 계산"
-> 기하가중평균: composite = Π(type_i)^w_i  (ln(1+composite%) = Σ w_i * ln(1+type_i%))

보고서에 가중치 수치 자체는 실려있지 않아, 사용자가 제공한 2026-07 실제 수치로 역산.
(전세/월세 두 개의 독립된 식 + 가중치합=1 제약으로 3개 미지수를 정확히 풂)
"""
import numpy as np
from scipy.optimize import fsolve

# 2026-07 전국주택가격동향조사 실제 발표치 (전월대비 상승률, %)
JEONSE = {"아파트": 0.49, "연립다세대": 0.27, "단독주택": 0.04, "종합": 0.38}
WOLSE = {"아파트": 0.47, "연립다세대": 0.31, "단독주택": 0.06, "종합": 0.38}


def equations(vars):
    w1, w2 = vars
    w3 = 1 - w1 - w2
    eq1 = (w1 * np.log(1 + JEONSE["아파트"] / 100) + w2 * np.log(1 + JEONSE["연립다세대"] / 100)
           + w3 * np.log(1 + JEONSE["단독주택"] / 100) - np.log(1 + JEONSE["종합"] / 100))
    eq2 = (w1 * np.log(1 + WOLSE["아파트"] / 100) + w2 * np.log(1 + WOLSE["연립다세대"] / 100)
           + w3 * np.log(1 + WOLSE["단독주택"] / 100) - np.log(1 + WOLSE["종합"] / 100))
    return [eq1, eq2]


def main():
    w1, w2 = fsolve(equations, [0.6, 0.25])
    w3 = 1 - w1 - w2
    print("역산된 유형별 가중치(주택재고량 기준 추정):")
    print(f"  아파트    : {w1*100:.2f}%")
    print(f"  연립다세대 : {w2*100:.2f}%")
    print(f"  단독주택   : {w3*100:.2f}%")

    for name, d in [("전세", JEONSE), ("월세", WOLSE)]:
        composite = np.exp(
            w1 * np.log(1 + d["아파트"] / 100)
            + w2 * np.log(1 + d["연립다세대"] / 100)
            + w3 * np.log(1 + d["단독주택"] / 100)
        )
        print(f"\n[{name}] 검산: 역산가중치로 재계산한 종합변동률 = {(composite-1)*100:.4f}%"
              f" (실제 발표치 {d['종합']}%)")

    return w1, w2, w3


if __name__ == "__main__":
    main()
