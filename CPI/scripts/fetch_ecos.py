"""
한국은행 ECOS Open API로 PPI(생산자물가지수 총지수, 월별)와 원/달러 환율(매매기준율, 일별->월평균)을
받아온다. 인증키는 저장소 밖 로컬 파일에서 읽는다(git에 올리지 않음).
"""
import json
from pathlib import Path

import pandas as pd
import requests

KEY_PATH = Path(r"C:\Users\infomax\Documents\DB for Claude\.credentials\ecos_api_key.txt")
CPI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS = CPI_DIR / "scripts"


def load_key() -> str:
    return KEY_PATH.read_text(encoding="utf-8").strip()


def fetch_statistic_search(key, stat_code, cycle, start, end, item_code, count=100000):
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/{count}/"
        f"{stat_code}/{cycle}/{start}/{end}/{item_code}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "StatisticSearch" not in data:
        raise RuntimeError(f"API error: {data}")
    rows = data["StatisticSearch"]["row"]
    return pd.DataFrame(rows)


def main():
    key = load_key()

    # 1) PPI 총지수 (월별, 2020=100)
    ppi = fetch_statistic_search(key, "404Y014", "M", "200001", "202612", "*AA")
    ppi = ppi[["TIME", "DATA_VALUE"]].rename(columns={"TIME": "기간", "DATA_VALUE": "PPI총지수(2020=100)"})
    ppi["기간"] = ppi["기간"].str[:4] + "-" + ppi["기간"].str[4:6]
    ppi["PPI총지수(2020=100)"] = pd.to_numeric(ppi["PPI총지수(2020=100)"])
    ppi.to_csv(SCRIPTS / "ecos_ppi_monthly.csv", index=False, encoding="utf-8-sig")
    print("PPI:", ppi.shape, ppi["기간"].min(), "~", ppi["기간"].max())

    # 2) 원/달러 환율 매매기준율 (일별 -> 월평균)
    fx = fetch_statistic_search(key, "731Y001", "D", "20000101", "20261231", "0000001")
    fx = fx[["TIME", "DATA_VALUE"]].rename(columns={"TIME": "일자", "DATA_VALUE": "원달러환율"})
    fx["원달러환율"] = pd.to_numeric(fx["원달러환율"])
    fx["기간"] = fx["일자"].str[:4] + "-" + fx["일자"].str[4:6]
    fx_monthly = fx.groupby("기간")["원달러환율"].mean().reset_index().rename(
        columns={"원달러환율": "원달러환율_월평균"}
    )
    fx.to_csv(SCRIPTS / "ecos_fx_daily.csv", index=False, encoding="utf-8-sig")
    fx_monthly.to_csv(SCRIPTS / "ecos_fx_monthly.csv", index=False, encoding="utf-8-sig")
    print("환율(일별):", fx.shape, fx["일자"].min(), "~", fx["일자"].max())
    print("환율(월평균):", fx_monthly.shape, fx_monthly["기간"].min(), "~", fx_monthly["기간"].max())

    print("\nPPI 최근 6개월:\n", ppi.tail(6).to_string(index=False))
    print("\n환율 최근 6개월:\n", fx_monthly.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
