"""
Tier A(일반시장가격형) 품목들의 최근 3개년(36개월) CPI 지수 데이터를 엑셀로 추출.
"""
import pandas as pd
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

panel = pd.read_csv(SCRIPTS / "all_tiers_monthly_panel.csv", encoding="utf-8-sig")
panel.columns = ["품목코드", "품목명", "가중치", "Tier", "분류근거", "비고"] + \
                 [c for c in panel.columns[6:]]

month_cols = sorted([c for c in panel.columns if c[:2] in ("19", "20")])
last36 = month_cols[-36:]
print(f"기간: {last36[0]} ~ {last36[-1]} ({len(last36)}개월)")

tier_a = panel[panel["Tier"] == "A"].copy()
print(f"Tier A 품목수: {len(tier_a)}")

out = tier_a[["품목코드", "품목명", "가중치"] + last36].sort_values("가중치", ascending=False)
out_path = SCRIPTS.parent / "TierA_품목_3개년데이터.xlsx"
out.to_excel(out_path, index=False, sheet_name="TierA_36개월")
print(f"저장: {out_path}")
