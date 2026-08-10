# CPI 월간 자동 업데이트 - Windows 작업 스케줄러에서 매일 실행
# (정확한 통계청 발표일이 매월 조금씩 달라서 매일 체크하는 방식으로 구현)

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$codeDir = "C:\Users\정우성\OneDrive\문서\DB for Claude\code"
Set-Location $codeDir

$py = "C:\Users\정우성\AppData\Local\Programs\Python\Python312\python.exe"
$log = Join-Path $codeDir "processed\kosis_monthly_update_log.txt"

$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $log -Value "===== $ts ====="

& $py update_kosis_monthly.py 2>&1 | Tee-Object -FilePath $log -Append

