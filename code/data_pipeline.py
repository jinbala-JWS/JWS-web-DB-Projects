"""
CPI 다음달 예측 프로그램 - 데이터 파이프라인

원본 파일들을 읽어서 모델링에 바로 쓸 수 있는 정제된 형태로 변환한다.

입력 (CPI 폴더):
  - 지출목적별_200001_201512.xls  (SpreadsheetML, 2000.01~2015.12 월별 지수)
  - 지출목적별_201601_202607.xls  (SpreadsheetML, 2016.01~2026.07 월별 지수)
  - CPI_Tier_분류.csv              (459개 품목의 Tier A/B/C/D 분류)
  - 지출목적별 품목 및 가중치(2000년-2022년).xlsx  (품목코드별 8개 시점 가중치)

입력 (외부데이터 폴더):
  - PPI_환율_200001_202607.csv
  - 오피넷_유가_200001_202607.csv

출력 (code/processed 폴더):
  - cpi_panel.csv       : 모든 지출목적별 행(대/중/소분류+개별품목) x 월별 지수, 롱포맷
  - item_master.csv     : Tier 분류 459개 품목의 코드/명/Tier/가중치 + cpi_panel과의 매칭여부
  - external_panel.csv  : PPI/환율/유가 데이터를 하나의 롱포맷으로 정리
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CPI_DIR = BASE_DIR / "CPI"
EXT_DIR = BASE_DIR / "외부데이터"
OUT_DIR = Path(__file__).resolve().parent / "processed"
OUT_DIR.mkdir(exist_ok=True)

SS_NS = "urn:schemas-microsoft-com:office:spreadsheet"


def _cell_text(cell):
    data = cell.find(f"{{{SS_NS}}}Data")
    if data is None:
        return None
    return data.text


def parse_spreadsheetml(path: Path) -> list[list[str | None]]:
    """SpreadsheetML(.xls로 저장된 XML) 파일을 행(row) 단위 셀 텍스트 리스트로 파싱.
    ss:Index를 존중해서 비어있는 셀도 위치를 맞춰 None으로 채운다."""
    raw = path.read_bytes().lstrip()  # 선행 탭/공백이 XML 선언 앞에 붙어있어 strict 파서가 거부함
    # 선언은 EUC-KR이지만 Python expat이 멀티바이트 CJK 인코딩 선언을 못 받아들이므로
    # 직접 cp949로 디코드한 뒤 UTF-8 선언으로 바꿔서 다시 인코딩한다.
    text = raw.decode("cp949", errors="replace")

    # 파일에는 "데이터" 시트 뒤에 "기타자료" 시트가 하나 더 있는데, 그 안의 텍스트에
    # 이스케이프 안 된 "<", ">"가 그대로 들어있어 XML 전체가 깨진다(예: "< 통계표 기타자료 >").
    # 필요한 건 첫 번째(데이터) 시트뿐이므로 그것만 잘라내서 파싱한다.
    start = text.index("<Worksheet")
    end = text.index("</Worksheet>", start) + len("</Worksheet>")
    fragment = text[start:end]
    wrapped = (
        f'<Workbook xmlns="{SS_NS}" xmlns:ss="{SS_NS}" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns:html="http://www.w3.org/TR/REC-html40" '
        'xmlns:o="urn:schemas-microsoft-com:office:office">' + fragment + "</Workbook>"
    )
    root = ET.fromstring(wrapped.encode("utf-8"))
    ws = root.find(f"{{{SS_NS}}}Worksheet")
    table = ws.find(f"{{{SS_NS}}}Table")
    rows_out = []
    for row in table.findall(f"{{{SS_NS}}}Row"):
        cells = row.findall(f"{{{SS_NS}}}Cell")
        row_vals: list[str | None] = []
        col_idx = 0
        for cell in cells:
            idx_attr = cell.get(f"{{{SS_NS}}}Index")
            if idx_attr is not None:
                target = int(idx_attr) - 1
                while col_idx < target:
                    row_vals.append(None)
                    col_idx += 1
            row_vals.append(_cell_text(cell))
            col_idx += 1
        rows_out.append(row_vals)
    return rows_out


MONTH_COL_RE = re.compile(r"^(\d{4})\.(\d{2})\s*월$")


def load_item_order(path: Path) -> list[str]:
    """파일에 등장하는 순서 그대로의 item_name_raw 리스트.
    KOSIS 원본은 트리 순회 순서(대분류->중분류->소분류->개별품목)로 행이 나열되어 있어서,
    이 순서를 그대로 이용해 계층구조(어느 대분류에 속하는지)를 복원한다."""
    rows = parse_spreadsheetml(path)
    order = []
    for row in rows[2:]:
        if not row or len(row) < 2:
            continue
        sido = row[0]
        item_raw = row[1]
        if not item_raw or sido != "전국":
            continue
        order.append(item_raw)
    return order


def build_item_hierarchy(item_order: list[str]) -> pd.DataFrame:
    """행 순서를 스캔하면서 "01 ...", "01.1 ..." 같은 숫자 코드 행을 만날 때마다
    현재 대분류/중분류를 갱신하고, 코드가 없는 leaf 품목에 매핑한다."""
    top_re = re.compile(r"^(\d{2})\s")          # 대분류: "01 식료품 및 비주류음료"
    mid_re = re.compile(r"^(\d{2}\.\d+)\s")      # 중분류: "01.1 식료품"
    sub_re = re.compile(r"^(\d{2}\.\d+\.\d+)\s")  # 소분류(있는 경우)

    cur_top = None
    cur_mid = None
    records = []
    for raw in item_order:
        name = raw.lstrip("　").strip()
        if name.startswith("0 총지수"):
            continue
        if top_re.match(name):
            cur_top = name
            cur_mid = None
            continue
        if sub_re.match(name):
            continue
        if mid_re.match(name):
            cur_mid = name
            continue
        # leaf item
        records.append({"item_name_raw": raw, "item_name": name,
                         "top_category": cur_top, "mid_category": cur_mid})
    df = pd.DataFrame(records)
    # "담배","서적"처럼 소분류명과 leaf 품목명이 동일해서 같은 행이 두 번 나열되는
    # 경우가 있음 -> 이름 기준 중복 제거(내용이 같으므로 안전)
    df = df.drop_duplicates(subset=["item_name"], keep="first")
    return df


def load_cpi_history(path: Path) -> pd.DataFrame:
    """지출목적별_*.xls 한 개 -> 롱포맷 DataFrame(item_name_raw, date, value)"""
    rows = parse_spreadsheetml(path)
    # header row is the 2nd row (index 1): 시도별, 지출목적별, 항목, 단위, <연도 컬럼...>, <월 컬럼...>
    header = rows[1]
    month_cols = {}  # col_idx -> "YYYY-MM"
    for i, col in enumerate(header):
        if not col:
            continue
        m = MONTH_COL_RE.match(col.strip())
        if m:
            month_cols[i] = f"{m.group(1)}-{m.group(2)}"

    records = []
    for row in rows[2:]:
        if not row or len(row) < 2:
            continue
        sido = row[0]
        item_raw = row[1]
        if not item_raw or sido != "전국":
            continue
        for col_idx, ym in month_cols.items():
            if col_idx >= len(row):
                continue
            val = row[col_idx]
            if val is None or val == "":
                continue
            try:
                fval = float(val)
            except ValueError:
                continue
            records.append((item_raw, ym, fval))

    return pd.DataFrame(records, columns=["item_name_raw", "date", "value"])


def build_cpi_panel() -> pd.DataFrame:
    f1 = CPI_DIR / "지출목적별_200001_201512.xls"
    f2 = CPI_DIR / "지출목적별_201601_202607.xls"
    df1 = load_cpi_history(f1)
    df2 = load_cpi_history(f2)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(subset=["item_name_raw", "date"], keep="last")

    # 들여쓰기 레벨(전각공백 갯수)과 stripped 품목명 추출
    def indent_level(name: str) -> int:
        stripped = name.lstrip("　")
        return (len(name) - len(stripped)) // 3  # 관찰상 레벨 하나당 전각공백 3개

    df["indent_level"] = df["item_name_raw"].apply(indent_level)
    df["item_name"] = df["item_name_raw"].str.lstrip("　").str.strip()

    df = df.sort_values(["item_name_raw", "date"]).reset_index(drop=True)
    return df


def load_tier_master() -> pd.DataFrame:
    path = CPI_DIR / "CPI_Tier_분류.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "품목코드": "item_code",
            "품목명": "item_name",
            "가중치": "weight_2022",
            "Tier": "tier",
            "분류근거": "tier_reason",
            "세부": "tier_detail",
        }
    )
    return df


def load_weight_history() -> pd.DataFrame:
    """가중치 8개 시점을 롱포맷으로: item_code, vintage_year, weight"""
    path = CPI_DIR / "지출목적별 품목 및 가중치(2000년-2022년).xlsx"
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    # 실측 구조(수기 확인): 열 순서 A..AB, 데이터는 6행부터.
    # B열=품목코드, D열=품목명, 가중치 8개 시점: G,J,M,P,S,V,Y,AB (0-based: 6,9,12,15,18,21,24,27)
    # 시점 연도는 헤더(5행, 0-based idx4)에서 텍스트로 확인해야 하나, 여기서는 알려진 순서를 그대로 사용:
    # 왼쪽부터 2022,2020,2017(또는2018),2015,2012,2010,2005,2000 순서일 가능성이 높음 -> 검증 필요.
    vintage_cols = {6: "V1", 9: "V2", 12: "V3", 15: "V4", 18: "V5", 21: "V6", 24: "V7", 27: "V8"}
    records = []
    for _, r in raw.iloc[5:].iterrows():
        code = r[1]
        name = r[3]
        if not isinstance(code, str) or not code.strip():
            continue
        for col_idx, vlabel in vintage_cols.items():
            val = r[col_idx] if col_idx in r.index else None
            if pd.isna(val):
                continue
            records.append((code.strip(), name, vlabel, float(val)))
    return pd.DataFrame(records, columns=["item_code", "item_name", "vintage", "weight"])


def load_external_panel() -> pd.DataFrame:
    ppi_fx = pd.read_csv(EXT_DIR / "PPI_환율_200001_202607.csv", encoding="utf-8-sig")
    oil = pd.read_csv(EXT_DIR / "오피넷_유가_200001_202607.csv", encoding="utf-8-sig")
    df = ppi_fx.merge(oil, on="년월", how="outer")
    df = df.rename(columns={"년월": "date"})
    return df.sort_values("date").reset_index(drop=True)


def main():
    print("1) CPI 히스토리 파싱 중...")
    panel = build_cpi_panel()
    panel.to_csv(OUT_DIR / "cpi_panel.csv", index=False, encoding="utf-8-sig")
    print(f"   -> {panel.shape[0]:,}행 저장 (품목/분류 {panel['item_name_raw'].nunique()}개, "
          f"기간 {panel['date'].min()}~{panel['date'].max()})")

    print("1-1) 계층구조(대/중분류) 복원 중...")
    item_order = load_item_order(CPI_DIR / "지출목적별_201601_202607.xls")
    hierarchy = build_item_hierarchy(item_order)
    hierarchy.to_csv(OUT_DIR / "item_hierarchy.csv", index=False, encoding="utf-8-sig")
    print(f"   -> leaf 품목 {hierarchy.shape[0]}개, 대분류 {hierarchy['top_category'].nunique()}개")

    print("2) Tier 분류표 로드 중...")
    tier = load_tier_master()
    tier.to_csv(OUT_DIR / "item_master.csv", index=False, encoding="utf-8-sig")
    print(f"   -> {tier.shape[0]}개 품목")

    print("3) 가중치 히스토리 로드 중...")
    try:
        weights = load_weight_history()
        weights.to_csv(OUT_DIR / "weight_history.csv", index=False, encoding="utf-8-sig")
        print(f"   -> {weights.shape[0]:,}행 (품목 {weights['item_code'].nunique()}개 x 시점)")
    except Exception as e:
        print(f"   !! 가중치 파일 파싱 실패(추후 수정 필요): {e}")

    print("4) 외부데이터(PPI/환율/유가) 병합 중...")
    ext = load_external_panel()
    ext.to_csv(OUT_DIR / "external_panel.csv", index=False, encoding="utf-8-sig")
    print(f"   -> {ext.shape[0]}행")

    print("5) Tier 품목 <-> CPI 패널 이름 매칭 점검 중...")
    leaf_names = set(panel["item_name"])
    tier_names = set(tier["item_name"])
    matched = tier_names & leaf_names
    unmatched = tier_names - leaf_names
    print(f"   -> 매칭 {len(matched)}/{len(tier_names)}개, 미매칭 {len(unmatched)}개")
    if unmatched:
        print("   미매칭 품목:", sorted(unmatched)[:30])

    # 이름 중복(동일 이름이 서로 다른 분류에 여러번 등장) 점검
    dup = panel.groupby("item_name")["item_name_raw"].nunique()
    dup = dup[dup > 1]
    if len(dup):
        print(f"   !! 이름 중복 품목 {len(dup)}개 (여러 raw name에 매핑됨):", list(dup.index[:20]))


if __name__ == "__main__":
    main()
