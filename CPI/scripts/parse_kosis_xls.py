"""
KOSIS 지출목적별 소비자물가지수 SpreadsheetML(.xls) 파서.
- 원본은 EUC-KR/CP949로 인코딩된 Excel 2003 XML(SpreadsheetML) 형식.
- '전국' 행만 추출해 (품목명 -> {YYYY-MM: 지수값}) 형태의 DataFrame으로 변환.
"""
import re
import pandas as pd
from pathlib import Path

CPI_DIR = Path(__file__).resolve().parent.parent


def parse_kosis_xls(path: Path) -> pd.DataFrame:
    with open(path, encoding="cp949") as f:
        data = f.read()

    row_blocks = re.findall(r"<Row.*?</Row>", data, re.S)

    def cells_of(row_block: str):
        # <Cell ...ss:Index="N"...><Data ...>value</Data></Cell> 형태 지원 (빈 셀은 Index로 건너뜀 가능)
        cells = []
        for m in re.finditer(r"<Cell([^>]*)>(?:<Data[^>]*>(.*?)</Data>)?</Cell>", row_block, re.S):
            attrs, val = m.group(1), m.group(2)
            idx_m = re.search(r'ss:Index="(\d+)"', attrs)
            if idx_m:
                idx = int(idx_m.group(1)) - 1
                while len(cells) < idx:
                    cells.append("")
            cells.append(val if val is not None else "")
        return cells

    header = cells_of(row_blocks[1])  # row0=제목, row1=헤더
    n_cols = len(header)

    # 헤더 컬럼 분류: 0~3 메타(시도별,지출목적별,항목,단위), 이후 연간(YYYY 년), 이후 월간(YYYY.MM 월)
    period_cols = []  # (col_index, period_label) period_label: 'YYYY' or 'YYYY-MM'
    for i in range(4, n_cols):
        h = header[i]
        m = re.match(r"(\d{4})\.(\d{2})\s*월", h)
        if m:
            period_cols.append((i, f"{m.group(1)}-{m.group(2)}"))
        else:
            m2 = re.match(r"(\d{4})\s*년", h)
            if m2:
                period_cols.append((i, m2.group(1)))  # 연간, 필요시 무시

    records = []
    for rb in row_blocks[2:]:
        c = cells_of(rb)
        if len(c) < 4:
            continue
        region = c[0]
        item = c[1]
        if region != "전국":
            continue
        row = {"품목": item}
        for i, period in period_cols:
            if len(period) == 7:  # 'YYYY-MM' 월간만 사용
                row[period] = c[i] if i < len(c) else ""
        records.append(row)

    df = pd.DataFrame(records)
    # 값 숫자 변환 (빈 문자열 -> NaN)
    month_cols = [c for c in df.columns if c != "품목"]
    for col in month_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


if __name__ == "__main__":
    df_old = parse_kosis_xls(CPI_DIR / "지출목적별_200001_201512.xls")
    df_new = parse_kosis_xls(CPI_DIR / "지출목적별_201601_202607.xls")

    print("old:", df_old.shape, "new:", df_new.shape)
    print("old 품목 수:", df_old["품목"].nunique(), "new 품목 수:", df_new["품목"].nunique())

    # 품목명 기준으로 두 파일 합치기 (월 컬럼이 겹치지 않아야 함)
    old_months = [c for c in df_old.columns if c != "품목"]
    new_months = [c for c in df_new.columns if c != "품목"]
    overlap = set(old_months) & set(new_months)
    print("겹치는 월 수:", len(overlap))

    merged = pd.merge(df_old, df_new, on="품목", how="outer")
    all_month_cols = sorted([c for c in merged.columns if c != "품목"])
    merged = merged[["품목"] + all_month_cols]
    out_path = CPI_DIR / "scripts" / "cpi_official_monthly_wide.csv"
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path, merged.shape)
