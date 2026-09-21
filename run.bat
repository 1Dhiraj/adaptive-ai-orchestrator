@echo off
setlocal
cd /d "%~dp0"

if /I "%~1"=="--check" goto check

set "ORCHESTRATOR_PORT=%~1"
if "%ORCHESTRATOR_PORT%"=="" set "ORCHESTRATOR_PORT=8000"

if not exist ".env.local" (
  echo ERROR: .env.local is missing. Copy .env.example to .env.local and configure NVIDIA_API_KEY.
  exit /b 1
)

python -c "from orchestrator.config import settings; print('Provider:', settings.resolve_provider()); print('Model:', settings.nvidia_model)"
if errorlevel 1 exit /b 1

echo Dashboard: http://127.0.0.1:%ORCHESTRATOR_PORT%/
python -m orchestrator.cli serve --port "%ORCHESTRATOR_PORT%"
exit /b %errorlevel%

:check
python -c "from orchestrator.config import settings; p=settings.resolve_provider(); print('Provider:', p); print('Model:', settings.nvidia_model); assert p == 'nvidia', 'LLM_PROVIDER is not nvidia'"
exit /b %errorlevel%
