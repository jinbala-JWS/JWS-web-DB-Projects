"""
CPI 월간 자동 업데이트 루틴.

매일 실행되는 것을 전제로(정확한 발표일이 매월 조금씩 다르므로), 실행할 때마다:
  1. KOSIS에서 "로컬에 있는 최신월 - 1개월" ~ "이번달"을 다시 받아온다
     (최신월 - 1개월도 다시 받는 이유: 통계청이 속보치를 확정치로 소폭 수정하는 경우가 있어서)
  2. cpi_panel.csv 에 병합(기존 자료 보존 + 새 자료/수정치 반영)
  3. 로컬 최신월이 실제로 앞으로 나갔으면(=새 달 발표됨) run_forecast.py 를 다시 돌려
     forecast_result_*.csv / forecast_summary.json 을 최신화한다
  4. 새 달이 아니면(아직 발표 전) 예측은 그대로 두고 조용히 종료

Windows 작업 스케줄러에서 매일 실행되도록 등록되어 있다(run_kosis_monthly.ps1 참고).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import collect_kosis_cpi as kosis
import run_forecast

CODE_DIR = Path(__file__).resolve().parent
PROC_DIR = CODE_DIR / "processed"
PANEL_PATH = PROC_DIR / "cpi_panel.csv"


def prev_month_ym(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[4:])
    m -= 1
    if m == 0:
        m = 12
        y -= 1
    return f"{y:04d}{m:02d}"


def main():
    if not PANEL_PATH.exists():
        print("cpi_panel.csv 가 없습니다. collect_kosis_cpi.py 를 먼저 전체 실행해주세요.")
        return

    existing = pd.read_csv(PANEL_PATH)
    old_max_date = existing["date"].max()  # "YYYY-MM"
    old_max_ym = old_max_date.replace("-", "")

    start_ym = prev_month_ym(old_max_ym)
    end_ym = kosis.default_end_ym()

    print(f"[체크] 로컬 최신월: {old_max_date} / 재조회 구간: {start_ym} ~ {end_ym}")

    key = kosis.load_api_key()
    meta = kosis.fetch_item_meta(key)
    hierarchy = kosis.build_hierarchy(meta)
    panel = kosis.fetch_all(key, start_ym, end_ym)

    if panel.empty:
        print("[결과] 새로 받아온 자료 없음(아직 미발표로 추정). 종료.")
        return

    panel_out = panel.copy()
    panel_out["item_name_raw"] = panel_out["item_name"]
    panel_out["indent_level"] = 0
    panel_out = panel_out[["item_name_raw", "date", "value", "indent_level", "item_name"]]

    combined = pd.concat([existing, panel_out], ignore_index=True)
    combined = combined.drop_duplicates(subset=["item_name", "date"], keep="last")
    combined = combined.sort_values(["item_name", "date"])
    combined.to_csv(PANEL_PATH, index=False, encoding="utf-8-sig")
    hierarchy.to_csv(PROC_DIR / "item_hierarchy.csv", index=False, encoding="utf-8-sig")

    new_max_date = combined["date"].max()
    print(f"[결과] 병합 완료. 최신월: {old_max_date} -> {new_max_date}")

    if new_max_date > old_max_date:
        print(f"[예측 갱신] 새 달({new_max_date}) 발표 감지. run_forecast 재실행...")
        run_forecast.run()
        print("[예측 갱신] 완료.")
    else:
        print("[예측 갱신] 새 달 없음(수정치만 반영됨). 예측 재실행 생략.")


if __name__ == "__main__":
    main()
