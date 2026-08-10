"""
KOSIS Open API로 "지출목적별 소비자물가지수"(DT_1J22001, 전국) 전체 이력을 받아온다.

기존에는 KOSIS 사이트에서 수동으로 내려받은 SpreadsheetML(xls) 파일 2개를 파싱했지만,
이제 공식 Open API로 대체한다. 2000-01부터 최신월까지, 전체 품목(대/중/소/세부분류 모두).

API 파라미터 규칙(리버스엔지니어링으로 확인됨, 공식 매뉴얼 문서와 실제 동작이 달랐음):
  - 엔드포인트는 반드시 /openapi/Param/statisticsParameterData.do 여야 함
    (매뉴얼 PDF에 나온 /openapi/statisticsData.do 는 이 테이블에서 계속 err=20 발생, 원인 불명)
  - itmId="T" 고정값(이 테이블의 "측정값=소비자물가지수" 축, 선택지가 하나뿐)
  - objL1="T10"  (시도별 축 - 전국)
  - objL2=<품목코드 or ALL>  (지출목적별 품목 축 - 실제 우리가 원하는 품목 분류)
  - jsonVD=Y 를 반드시 넣어야 응답이 정상 JSON(키에 큰따옴표)으로 옴. 빠지면 키가
    따옴표 없는 JS 객체 리터럴 형태로 와서 json.loads가 깨짐.
  - 한 번에 최대 40,000행. 품목 581개 x 60개월 = 34,860행으로 청크를 나눠서 요청.

사용법:
    python collect_kosis_cpi.py                       # 2000-01 ~ 이번달까지 전체 재수집
    python collect_kosis_cpi.py --start 202601 --end 202607   # 최신월만 갱신(자동수집용)
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

CODE_DIR = Path(__file__).resolve().parent
PROC_DIR = CODE_DIR / "processed"
API_KEY_PATH = CODE_DIR / "kosis_api_key.txt"

BASE_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
META_URL = "https://kosis.kr/openapi/statisticsData.do"
ORG_ID = "101"
TBL_ID = "DT_1J22001"

CHUNK_MONTHS = 60  # 581개 품목 x 60개월 = 34,860행 (40,000행 제한 이내)


def load_api_key() -> str:
    if not API_KEY_PATH.exists():
        raise RuntimeError(
            f"{API_KEY_PATH} 파일이 없습니다. KOSIS Open API 인증키를 이 파일에 한 줄로 저장해주세요."
        )
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(f"{API_KEY_PATH} 가 비어있습니다.")
    return key


def month_chunks(start_ym: str, end_ym: str, size: int = CHUNK_MONTHS):
    """'YYYYMM' 문자열 구간을 size개월씩 나눠 (chunk_start, chunk_end) 튜플로 yield."""
    sy, sm = int(start_ym[:4]), int(start_ym[4:])
    ey, em = int(end_ym[:4]), int(end_ym[4:])

    cy, cm = sy, sm
    while (cy, cm) <= (ey, em):
        chunk_start = f"{cy:04d}{cm:02d}"
        # size개월 뒤로 이동
        ny, nm = cy, cm + size - 1
        while nm > 12:
            nm -= 12
            ny += 1
        if (ny, nm) > (ey, em):
            ny, nm = ey, em
        chunk_end = f"{ny:04d}{nm:02d}"
        yield chunk_start, chunk_end

        cy, cm = ny, nm + 1
        while cm > 12:
            cm -= 12
            cy += 1


def fetch_item_meta(key: str) -> pd.DataFrame:
    """품목 분류 메타(ITM_ID, ITM_NM, UP_ITM_ID) - 공식 위계구조를 그대로 가져온다."""
    params = {
        "method": "getMeta",
        "apiKey": key,
        "type": "ITM",
        "orgId": ORG_ID,
        "tblId": TBL_ID,
        "format": "json",
        "jsonVD": "Y",
    }
    r = requests.get(META_URL, params=params, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    data = json.loads(r.text)
    df = pd.DataFrame(data)
    # OBJ_ID == "I" 인 것만 "지출목적별" 품목 축(우리가 원하는 것). "C"(시도별), "ITEM"(측정값)은 제외.
    df = df[df["OBJ_ID"] == "I"].copy()
    return df[["ITM_ID", "ITM_NM", "UP_ITM_ID"]].drop_duplicates(subset="ITM_ID")


def build_hierarchy(meta: pd.DataFrame) -> pd.DataFrame:
    """각 품목의 최상위(대분류) 조상 이름을 찾아 item_hierarchy.csv 형태로 만든다."""
    id_to_name = dict(zip(meta["ITM_ID"], meta["ITM_NM"]))
    id_to_parent = dict(zip(meta["ITM_ID"], meta["UP_ITM_ID"]))

    def top_ancestor(item_id: str) -> str | None:
        seen = set()
        cur = item_id
        while True:
            parent = id_to_parent.get(cur)
            if not isinstance(parent, str) or not parent or parent in seen:
                break
            seen.add(parent)
            cur = parent
        return cur if cur != "0" else None

    def mid_ancestor(item_id: str) -> str | None:
        """대분류 바로 아래(중분류) 조상. top_ancestor의 직계 자식을 역으로 추적."""
        top = top_ancestor(item_id)
        if top is None:
            return None
        cur = item_id
        prev = None
        seen = set()
        while True:
            parent = id_to_parent.get(cur)
            if not isinstance(parent, str) or not parent or parent in seen:
                break
            if parent == top:
                return id_to_name.get(prev) if prev else id_to_name.get(cur)
            seen.add(parent)
            prev = cur
            cur = parent
        return None

    rows = []
    for item_id, name in id_to_name.items():
        top = top_ancestor(item_id)
        mid = mid_ancestor(item_id)
        rows.append({
            "item_id": item_id,
            "item_name_raw": name,
            "item_name": name,
            "top_category": id_to_name.get(top) if top else None,
            "mid_category": mid,
            "depth": len(item_id) if item_id != "0" else 0,
        })
    hdf = pd.DataFrame(rows)
    # 이름 충돌(예: "담배"가 소분류명과 세부품목명에 동시 사용) 시,
    # item_id가 더 긴(더 깊은=실제 낱개 품목) 쪽을 우선해서 남긴다.
    hdf = hdf.sort_values("depth", ascending=False).drop_duplicates(subset="item_name", keep="first")
    return hdf[["item_name_raw", "item_name", "top_category", "mid_category"]]


def fetch_chunk(key: str, start_ym: str, end_ym: str) -> list[dict]:
    params = {
        "method": "getList",
        "apiKey": key,
        "orgId": ORG_ID,
        "tblId": TBL_ID,
        "itmId": "T",
        "objL1": "T10",
        "objL2": "ALL",
        "prdSe": "M",
        "startPrdDe": start_ym,
        "endPrdDe": end_ym,
        "format": "json",
        "jsonVD": "Y",
    }
    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    r.encoding = "utf-8"
    data = json.loads(r.text)
    if isinstance(data, dict):
        raise RuntimeError(f"KOSIS API 오류 ({start_ym}~{end_ym}): {data}")
    return data


def default_end_ym() -> str:
    """당월 데이터는 아직 없으므로 당월 문자열을 반환(API가 알아서 있는 만큼만 줌)."""
    today = date.today()
    return f"{today.year:04d}{today.month:02d}"


def fetch_all(key: str, start_ym: str, end_ym: str) -> pd.DataFrame:
    all_rows = []
    chunks = list(month_chunks(start_ym, end_ym))
    for i, (cs, ce) in enumerate(chunks, 1):
        rows = fetch_chunk(key, cs, ce)
        all_rows.extend(rows)
        print(f"[{i}/{len(chunks)}] {cs}~{ce}: {len(rows)}행 수집")
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df = df[["C2", "C2_NM", "PRD_DE", "DT"]].rename(
        columns={"C2": "item_id", "C2_NM": "item_name", "PRD_DE": "date", "DT": "value"}
    )
    df["date"] = df["date"].str[:4] + "-" + df["date"].str[4:]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.drop_duplicates(subset=["item_name", "date"], keep="last")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="200001", help="YYYYMM, 기본 200001")
    parser.add_argument("--end", default=None, help="YYYYMM, 기본 이번달")
    parser.add_argument("--full-rebuild", action="store_true",
                         help="지정하면 processed/cpi_panel.csv, item_hierarchy.csv 를 통째로 새로 씀(기본 동작)")
    args = parser.parse_args()

    key = load_api_key()
    end_ym = args.end or default_end_ym()

    print("=== 품목 메타(위계구조) 조회 ===")
    meta = fetch_item_meta(key)
    hierarchy = build_hierarchy(meta)
    print(f"품목 {len(meta)}개, 이름충돌 제거 후 {len(hierarchy)}개")

    print(f"\n=== 데이터 조회: {args.start} ~ {end_ym} ===")
    panel = fetch_all(key, args.start, end_ym)
    print(f"\n총 {len(panel)}행 수집 완료 (품목 {panel['item_name'].nunique()}개, "
          f"기간 {panel['date'].min()} ~ {panel['date'].max()})")

    PROC_DIR.mkdir(exist_ok=True)

    panel_out = panel.copy()
    panel_out["item_name_raw"] = panel_out["item_name"]
    panel_out["indent_level"] = 0
    panel_out = panel_out[["item_name_raw", "date", "value", "indent_level", "item_name"]]

    panel_path = PROC_DIR / "cpi_panel.csv"
    if panel_path.exists() and not args.full_rebuild:
        # 기존 자료와 병합: 새로 받아온 (품목,월) 조합이 있으면 그걸로 덮어쓰고(수정치 반영),
        # 없던 기간/품목만 새로 추가한다. 이번에 조회 범위 밖의 과거 데이터는 그대로 보존.
        existing = pd.read_csv(panel_path)
        combined = pd.concat([existing, panel_out], ignore_index=True)
        combined = combined.drop_duplicates(subset=["item_name", "date"], keep="last")
        combined = combined.sort_values(["item_name", "date"])
        combined.to_csv(panel_path, index=False, encoding="utf-8-sig")
        print(f"\n병합 저장 완료: {panel_path} (전체 {len(combined)}행, 이번 조회분 {len(panel_out)}행)")
    else:
        panel_out.to_csv(panel_path, index=False, encoding="utf-8-sig")
        print(f"\n저장 완료: {panel_path} ({len(panel_out)}행)")

    hierarchy.to_csv(PROC_DIR / "item_hierarchy.csv", index=False, encoding="utf-8-sig")
    print(f"저장 완료: {PROC_DIR / 'item_hierarchy.csv'}")


if __name__ == "__main__":
    main()
