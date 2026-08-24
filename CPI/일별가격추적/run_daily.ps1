# 매일 실행: 가격 수집 -> (변경 있으면) git add/commit/push
# Windows 작업 스케줄러가 이 스크립트를 매일 호출한다.

$ErrorActionPreference = "Continue"
$repoRoot = "C:\Users\infomax\Documents\DB for Claude\JWS-web-DB-Projects"
$scriptDir = "$repoRoot\CPI\일별가격추적"
$logFile = "$scriptDir\run_log.txt"

function Write-Log($msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $msg"
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

Write-Log "=== 실행 시작 ==="

Set-Location $scriptDir
$pyOutput = & python daily_price_collector.py 2>&1
Write-Log ($pyOutput -join " | ")

Set-Location $repoRoot
$status = git status --porcelain -- "CPI/일별가격추적"
if ($status) {
    git add "CPI/일별가격추적"
    git commit -m "chore: 일별 가격데이터 자동수집 $(Get-Date -Format 'yyyy-MM-dd')

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>" | Out-Null
    git push | Out-Null
    Write-Log "git commit/push 완료"
} else {
    Write-Log "변경 없음 (이미 수집됨) - git 스킵"
}

Write-Log "=== 실행 종료 ==="
