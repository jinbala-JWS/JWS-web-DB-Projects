"""
CPI 다음달 예측 - 메인 실행 스크립트

processed/ 폴더의 정제된 데이터를 읽어서:
  1. 458개 개별품목을 Tier(A/B/C/D)별 방식으로 예측
  2. 공식 가중치(2022년 기준)로 라스파이레스 가중합 -> 대분류 12개 + 총지수 예측
  3. 검증: "예측방식 그대로 지난달 실측치를 재구성"해서 공식 대분류/총지수와 비교(방법론 정합성 체크)
  4. 결과를 콘솔 출력 + processed/forecast_result.csv 로 저장
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import models

CODE_DIR = Path(__file__).resolve().parent
PROC_DIR = CODE_DIR / "processed"


def load_data():
    panel = pd.read_csv(PROC_DIR / "cpi_panel.csv")
    item_master = pd.read_csv(PROC_DIR / "item_master.csv")
    hierarchy = pd.read_csv(PROC_DIR / "item_hierarchy.csv")
    external = pd.read_csv(PROC_DIR / "external_panel.csv")
    kapt_path = PROC_DIR / "kapt_management_fee_monthly.csv"
    kapt = pd.read_csv(kapt_path) if kapt_path.exists() else pd.DataFrame(columns=["date"])
    return panel, item_master, hierarchy, external, kapt


def pivot_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """item_name x date 와이드 포맷. 같은 이름이 여러 raw name에 걸리는 경우는 없음(사전 점검됨)."""
    wide = panel.pivot_table(index="date", columns="item_name", values="value", aggfunc="last")
    wide = wide.sort_index()
    return wide


def compute_parent_trend(wide: pd.DataFrame, top_category: str | None, n: int = 3) -> float | None:
    if not top_category or top_category not in wide.columns:
        return None
    s = wide[top_category].dropna()
    mom = s.pct_change().dropna()
    if len(mom) < n:
        return None
    return float(mom.iloc[-n:].mean())


def group_from_detail(detail: str) -> str | None:
    if not isinstance(detail, str) or not detail:
        return None
    return detail.split("/")[0].strip()


def run(target_override_month: str | None = None, verbose: bool = True):
    panel, item_master, hierarchy, external, kapt = load_data()
    wide = pivot_panel(panel)
    external = external.set_index("date")
    kapt_wide = kapt.set_index("date") if len(kapt) else kapt

    merged = item_master.merge(hierarchy[["item_name", "top_category"]], on="item_name", how="left")
    overrides = models.load_overrides()

    last_month = wide.index.max()
    target_ym = target_override_month or models.next_month_str(last_month)

    results = []

    # ---- Tier B 1차 패스: 비고정계절 품목(또는 출하기간인 고정계절 품목) 먼저 예측해서
    #      신선과실/신선채소/신선어개 그룹별 평균 모멘텀을 구한다 ----
    tier_b_rows = merged[merged["tier"] == "B"]
    b_normal_mom_by_group: dict[str, list[float]] = {}
    b_prelim = {}
    for _, row in tier_b_rows.iterrows():
        name = row["item_name"]
        if name not in wide.columns:
            continue
        hist = wide[name]
        detail = row.get("tier_detail", "")
        unavailable = models.parse_unavailable_months(detail)
        target_m = models.target_month_number(target_ym)
        is_fixed_and_out = bool(unavailable and target_m in unavailable)
        if not is_fixed_and_out:
            res = models.forecast_tier_b_normal(hist)
            b_prelim[name] = res
            grp = group_from_detail(detail) or "기타"
            if "mom_pct" in res and np.isfinite(res["mom_pct"]):
                b_normal_mom_by_group.setdefault(grp, []).append(res["mom_pct"] / 100)

    group_fallback_mom = {g: float(np.mean(v)) for g, v in b_normal_mom_by_group.items() if v}

    # ---- 전체 품목 예측 ----
    for _, row in merged.iterrows():
        name = row["item_name"]
        code = row["item_code"]
        tier = row["tier"]
        weight = row["weight_2022"]
        detail = row.get("tier_detail", "")

        if name not in wide.columns:
            results.append({"item_code": code, "item_name": name, "tier": tier,
                             "weight": weight, "forecast": np.nan, "method": "no_history",
                             "mom_pct": np.nan})
            continue

        hist = wide[name]

        if tier == "A" and name in models.TREND_ANCHOR_ITEMS:
            ext_col = models.TREND_ANCHOR_ITEMS[name]
            ext_hist = kapt_wide[ext_col] if (len(kapt_wide) and ext_col in kapt_wide.columns) else None
            res = models.forecast_seasonal_plus_trend(hist, ext_hist)
        elif tier == "A":
            parent_mom = compute_parent_trend(wide, row.get("top_category"))
            res = models.forecast_tier_a(hist, parent_trend_mom=parent_mom)
        elif tier == "B":
            if name in b_prelim:
                res = b_prelim[name]
            else:
                grp = group_from_detail(detail) or "기타"
                fallback = group_fallback_mom.get(grp)
                res = models.forecast_tier_b(hist, detail, target_ym, fallback)
        elif tier == "C":
            res = models.forecast_tier_c(hist, code, overrides)
        elif tier == "D":
            res = models.forecast_tier_d(name, hist, external.reset_index(), code, overrides)
        else:
            res = {"forecast": hist.dropna().iloc[-1], "method": "unknown_tier", "mom_pct": 0.0}

        results.append({
            "item_code": code, "item_name": name, "tier": tier, "weight": weight,
            "forecast": res.get("forecast"), "method": res.get("method"),
            "mom_pct": res.get("mom_pct"),
        })

    res_df = pd.DataFrame(results)

    # ---- 라스파이레스 가중합으로 대분류/총지수 재구성 ----
    valid = res_df.dropna(subset=["forecast", "weight"])
    total_index_forecast = float((valid["forecast"] * valid["weight"]).sum() / valid["weight"].sum())

    merged_top = merged[["item_code", "top_category"]]
    res_with_top = res_df.merge(merged_top, on="item_code", how="left")
    by_top = (
        res_with_top.dropna(subset=["forecast", "weight"])
        .groupby("top_category")
        .apply(lambda g: pd.Series({
            "forecast_index": (g["forecast"] * g["weight"]).sum() / g["weight"].sum(),
            "n_items": len(g),
            "weight_sum": g["weight"].sum(),
        }))
        .reset_index()
    )

    # ---- 방법론 검증: 같은 가중합 방식으로 "지난달"을 실측치로 재구성해서 공식 총지수와 비교 ----
    last_actual_total = wide["0 총지수"].iloc[-1] if "0 총지수" in wide.columns else None
    last_vals = merged.merge(pd.DataFrame({"item_name": wide.columns, "_dummy": 0}), on="item_name", how="left")
    check_rows = []
    for _, row in merged.iterrows():
        name = row["item_name"]
        if name in wide.columns:
            v = wide[name].iloc[-1]
            check_rows.append((row["weight_2022"], v))
    if check_rows:
        w = np.array([r[0] for r in check_rows])
        v = np.array([r[1] for r in check_rows])
        reconstructed_last = float((w * v).sum() / w.sum())
    else:
        reconstructed_last = None

    prev_actual_total = wide["0 총지수"].iloc[-2] if "0 총지수" in wide.columns else None
    forecast_mom_pct = (total_index_forecast / last_actual_total - 1) * 100 if last_actual_total else None

    summary = {
        "last_month": last_month,
        "target_month": target_ym,
        "last_actual_total_index": last_actual_total,
        "reconstructed_last_month_index(방법론검증)": reconstructed_last,
        "reconstruction_error_pct": (
            (reconstructed_last / last_actual_total - 1) * 100
            if reconstructed_last and last_actual_total else None
        ),
        "forecast_total_index": total_index_forecast,
        "forecast_mom_pct": forecast_mom_pct,
    }

    if verbose:
        print("=" * 60)
        print(f"[검증] {last_month} 실측 총지수: {last_actual_total:.2f}")
        print(f"[검증] 같은 가중합 방식으로 재구성한 {last_month} 총지수: {reconstructed_last:.2f} "
              f"(오차 {summary['reconstruction_error_pct']:+.3f}%)")
        print("-" * 60)
        print(f"[예측] {target_ym} 총지수 예측치: {total_index_forecast:.2f} "
              f"(전월비 {forecast_mom_pct:+.3f}%)")
        print("=" * 60)
        print("\n[대분류별 예측]")
        print(by_top.sort_values("top_category").to_string(index=False))
        print("\n[Tier별 품목 수 / 평균 전월비%]")
        print(res_df.groupby("tier").agg(n=("item_code", "count"),
                                          avg_mom=("mom_pct", "mean")).round(3))

    res_df.to_csv(PROC_DIR / "forecast_result_items.csv", index=False, encoding="utf-8-sig")
    by_top.to_csv(PROC_DIR / "forecast_result_categories.csv", index=False, encoding="utf-8-sig")
    with open(PROC_DIR / "forecast_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return summary, res_df, by_top


if __name__ == "__main__":
    run()
