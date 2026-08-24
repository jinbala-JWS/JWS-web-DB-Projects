# CPI_DailyPriceCollector 작업 스케줄러 등록 스크립트 (재현/재등록용).
# 이미 한 번 등록되어 있으면 -Force로 덮어씀. 관리자 권한 불필요(현재 로그인 사용자 계정으로 실행).
#
# 실행: powershell -ExecutionPolicy Bypass -File register_daily_task.ps1

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument '-ExecutionPolicy Bypass -File "C:\Users\infomax\Documents\DB for Claude\JWS-web-DB-Projects\CPI\일별가격추적\run_daily.ps1"'
$trigger = New-ScheduledTaskTrigger -Daily -At 08:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName "CPI_DailyPriceCollector" `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "매일 오피넷 유가/환율 수집 후 저장소에 자동 커밋·푸시" -Force

Get-ScheduledTask -TaskName "CPI_DailyPriceCollector" | Select-Object TaskName, State
