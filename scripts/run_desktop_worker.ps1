$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Desktop worker must run as a standard Windows user, not Administrator."
}
if (-not $env:ORCHESTRATOR_STATE_URL -or -not $env:ORCHESTRATOR_BROKER_URL) {
    throw "Set ORCHESTRATOR_STATE_URL and ORCHESTRATOR_BROKER_URL first."
}
if (-not (Get-Command hermes -ErrorAction SilentlyContinue)) {
    throw "Hermes is not installed or is not on PATH."
}
hermes computer-use doctor
if ($LASTEXITCODE -ne 0) { throw "Hermes computer-use doctor failed." }
$env:ORCHESTRATOR_ALLOW_DESKTOP = "true"
python -m celery -A orchestrator.distributed:app worker --loglevel=INFO -Q desktop --concurrency=1

