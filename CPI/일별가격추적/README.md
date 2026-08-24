# 일별 가격데이터 자동수집 (2026-08-24 구축)

매일 자동으로 유가·환율을 받아 [daily_price_log.csv](./daily_price_log.csv)에 한 줄씩 쌓는다.

## 구성

| 파일 | 역할 |
|---|---|
| [daily_price_collector.py](./daily_price_collector.py) | 오피넷(휘발유·경유·등유·자동차용LPG)·ECOS(원달러환율) 오늘자 값을 받아 CSV에 append. 이미 오늘 날짜가 있으면 스킵(재실행 안전) |
| [run_daily.ps1](./run_daily.ps1) | 수집 스크립트 실행 → 변경 있으면 git add/commit/push까지 자동 처리. [run_log.txt](./run_log.txt)에 실행이력 기록 |
| [register_daily_task.ps1](./register_daily_task.ps1) | Windows 작업 스케줄러에 매일 12:00 실행되도록 등록(재현용) |
| [daily_price_log.csv](./daily_price_log.csv) | 실제 누적 데이터 |

## 동작 방식

- 브라우저 없이 순수 `requests`로 오피넷 폼 제출을 재현(POST 파라미터를 직접 구성) — 오피넷 페이지가
  구간별로 인코딩이 섞여있어(정적 라벨=CP949, 결과테이블=UTF-8) 결과 테이블은 UTF-8로 디코딩.
- Windows 작업 스케줄러(`CPI_DailyPriceCollector`, 매일 12:00, 로그인 사용자 계정으로 실행 —
  관리자 권한·비밀번호 저장 불필요, 단 그 시각에 로그인 상태여야 실행됨)가 매일 `run_daily.ps1`을 호출.
- 오피넷 데이터는 통상 전일자 기준(발표 지연)이라 `오피넷_기준일` 컬럼으로 실제 기준일을 별도 기록.

## 확장 시 참고

- 오피넷 폼 파라미터(POST 필드명)는 브라우저 개발자도구 없이 `new FormData(document.forms['form1'])`로
  Claude Browser의 javascript_tool을 통해 추출함 — 다른 오피넷 페이지(예: 지역별/상표별)를
  추가할 때도 동일한 방법 사용 가능.
- 재등록/시간 변경: `register_daily_task.ps1`의 `-At 12:00` 수정 후 재실행.
- 작업 확인: `Get-ScheduledTask -TaskName "CPI_DailyPriceCollector"` 또는 Windows "작업 스케줄러" GUI.
