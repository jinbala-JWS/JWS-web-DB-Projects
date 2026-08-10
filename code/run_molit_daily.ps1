# 매일 국토교통부 실거래가 전월세 데이터를 하루치 한도만큼 이어받는 스케줄 작업용 래퍼.
# Windows 작업 스케줄러("MOLIT_Rental_Collector")가 매일 이 스크립트를 실행한다.

$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
Set-Location "C:\Users\정우성\OneDrive\문서\DB for Claude\code"

$logFile = "C:\Users\정우성\OneDrive\문서\DB for Claude\code\processed\molit_collect_log.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logFile -Value "`n=== $timestamp ===" -Encoding UTF8

# 1순위: 최근 1년치 원자료(11~10일 재집계용, 계약일자 보존) 확보 - 한번 다 받으면 더는 요청 안 함
python collect_molit_raw_recent.py --months 14 --types A,B,C --max-requests 45 2>&1 |
    Tee-Object -FilePath $logFile -Append

# 2순위: 남은 한도로 2011년부터의 옛날 달 백필 이어가기
python collect_molit_rental.py --start 2011-01 --end 2026-07 --types A,B,C --max-requests 45 2>&1 |
    Tee-Object -FilePath $logFile -Append

# 전 구간(561건)이 모두 채워졌으면 더 이상 돌 필요 없으니 스케줄 작업을 스스로 비활성화한다.
$csvPath = "C:\Users\정우성\OneDrive\문서\DB for Claude\code\processed\molit_rental_monthly.csv"
if (Test-Path $csvPath) {
    $rowCount = (Import-Csv $csvPath).Count
    if ($rowCount -ge 561) {
        Add-Content -Path $logFile -Value "561건 모두 수집 완료 - 스케줄 작업 비활성화" -Encoding UTF8
        schtasks /change /tn "MOLIT_Rental_Collector" /disable | Out-Null
    }
}



