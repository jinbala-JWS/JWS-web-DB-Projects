"""
매일 실행되는 유가/환율 일별 수집기.
브라우저 없이 순수 requests로 오피넷 폼을 재현해 오늘자 데이터를 받아
daily_price_log.csv 에 한 줄씩 누적한다(같은 날짜가 이미 있으면 스킵, 재실행해도 안전).

Windows 작업 스케줄러에서 매일 이 스크립트를 실행하도록 등록되어 있다.
(등록: register_daily_task.ps1 참조)
"""
import csv
import sys
from datetime import date
from pathlib import Path

import requests

# Windows 작업 스케줄러 실행 시 콘솔 코드페이지(cp949)가 한글 출력에서 깨지는 것 방지
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "daily_price_log.csv"
KEY_PATH = Path(r"C:\Users\infomax\Documents\DB for Claude\.credentials\ecos_api_key.txt")

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_oil_prices(today: date) -> dict:
    """오피넷 주유소 평균판매가격(일간) - 고급휘발유/보통휘발유/자동차용경유/실내등유."""
    url = "https://www.opinet.co.kr/user/dopospdrg/dopOsPdrgSelect.do"
    y, m, d = f"{today.year}", f"{today.month:02d}", f"{today.day:02d}"
    data = {
        "all_chk_cnt": "5", "INIF_FLAG": "N", "chk_cnt": "4",
        "sta_dt": "", "end_dt": "",
        "TERM": "D", "STA_Y": y, "STA_M": m, "STA_D": d,
        "END_Y": y, "END_M": m, "END_D": d,
        "OIL_CD_B034": "Y", "OIL_CD_B027": "Y", "OIL_CD_D047": "Y", "OIL_CD_C004": "Y",
        "equal": "Y",
    }
    s = requests.Session()
    s.get(url, headers=HEADERS, timeout=20)
    r = s.post(url, data=data, headers={**HEADERS, "Referer": url}, timeout=20)
    # 이 페이지는 인코딩이 섞여있음(정적 라벨=CP949, AJAX 결과테이블=UTF-8) -
    # 우리가 필요한 결과 테이블(numbox)은 실측 결과 UTF-8이라 utf-8로 디코딩
    r.encoding = "utf-8"
    text = r.text

    # <tbody id="numbox"> 안의 첫 번째 <tr> 한 행(가장 최근 일자)을 파싱
    import re
    m_body = re.search(r'id="numbox"', text)
    if not m_body:
        raise RuntimeError("오피넷 응답에서 결과 테이블(numbox)을 찾지 못함")
    tail = text[m_body.end():]
    row_match = re.search(r"<tr[^>]*>(.*?)</tr>", tail, re.S)
    if not row_match:
        raise RuntimeError("오피넷 응답에서 데이터 행을 찾지 못함")
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row_match.group(1), re.S)
    cells = [c.strip() for c in cells if c.strip()]
    if len(cells) < 5:
        raise RuntimeError(f"오피넷 데이터 행 파싱 실패: {cells}")
    # cells[0] = 날짜(예: 2026년08월22일), 1=고급휘발유, 2=보통휘발유, 3=자동차용경유, 4=실내등유
    def num(s):
        s = s.replace(",", "").strip()
        return float(s) if s and s != "-" else None

    return {
        "오피넷_기준일": cells[0],
        "고급휘발유": num(cells[1]),
        "보통휘발유": num(cells[2]),
        "자동차용경유": num(cells[3]),
        "실내등유": num(cells[4]),
    }


def fetch_auto_lpg(today: date) -> dict:
    """오피넷 자동차충전소(자동차부탄) 일간 평균가."""
    url = "https://www.opinet.co.kr/user/dopvsavsel/dopVsAvselSelect.do"
    y, m, d = f"{today.year}", f"{today.month:02d}", f"{today.day:02d}"
    data = {
        "TERM": "D", "STA_Y": y, "STA_M": m, "STA_D": d, "STA_W": "1", "STA_B": "1",
        "END_Y": y, "END_M": m, "END_D": d, "END_W": "1", "END_B": "1",
        "OIL_CD_K015_P": "Y", "OIL_CD_K021_P": "", "OIL_CD_K022_P": "",
    }
    s = requests.Session()
    s.get(url, headers=HEADERS, timeout=20)
    r = s.post(url, data=data, headers={**HEADERS, "Referer": url}, timeout=20)
    r.encoding = "utf-8"
    text = r.text

    import re
    # 이 페이지는 결과 테이블 id가 "tbody1"(휘발유/경유 페이지의 "numbox"와 다름)
    m_body = re.search(r'id="tbody1"', text)
    if not m_body:
        raise RuntimeError("오피넷(LPG) 응답에서 결과 테이블을 찾지 못함")
    tail = text[m_body.end():]
    row_match = re.search(r"<tr[^>]*>(.*?)</tr>", tail, re.S)
    if not row_match:
        raise RuntimeError("오피넷(LPG) 응답에서 데이터 행을 찾지 못함")
    cells = re.findall(r"<td[^>]*>(.*?)</td>", row_match.group(1), re.S)
    cells = [c.strip() for c in cells if c.strip()]
    if len(cells) < 2:
        raise RuntimeError(f"오피넷(LPG) 데이터 행 파싱 실패: {cells}")

    def num(s):
        s = s.replace(",", "").strip()
        return float(s) if s and s != "-" else None

    return {"자동차용LPG": num(cells[1])}


def fetch_fx_today(today: date) -> dict:
    """ECOS 원/달러 매매기준율(일별) - 오늘 값이 아직 없으면 최근 영업일 값."""
    if not KEY_PATH.exists():
        return {"원달러환율": None}
    key = KEY_PATH.read_text(encoding="utf-8").strip()
    d = today.strftime("%Y%m%d")
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/5/731Y001/D/{d}/{d}/0000001"
    try:
        r = requests.get(url, timeout=20)
        data = r.json()
        rows = data.get("StatisticSearch", {}).get("row", [])
        if rows:
            return {"원달러환율": float(rows[-1]["DATA_VALUE"])}
    except Exception:
        pass
    return {"원달러환율": None}


def main():
    today = date.today()
    today_str = today.isoformat()

    existing_dates = set()
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                existing_dates.add(row["수집일자"])

    if today_str in existing_dates:
        print(f"{today_str} 는 이미 수집됨 - 스킵")
        return

    record = {"수집일자": today_str}
    errors = []
    try:
        record.update(fetch_oil_prices(today))
    except Exception as e:
        errors.append(f"유가: {e}")
    try:
        record.update(fetch_auto_lpg(today))
    except Exception as e:
        errors.append(f"자동차용LPG: {e}")
    try:
        record.update(fetch_fx_today(today))
    except Exception as e:
        errors.append(f"환율: {e}")

    fieldnames = ["수집일자", "오피넷_기준일", "고급휘발유", "보통휘발유", "자동차용경유",
                  "실내등유", "자동차용LPG", "원달러환율"]
    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fieldnames})

    print(f"{today_str} 저장 완료:", {k: record.get(k) for k in fieldnames})
    if errors:
        print("일부 실패:", "; ".join(errors), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
