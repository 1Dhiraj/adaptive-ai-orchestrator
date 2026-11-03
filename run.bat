@echo off
setlocal
cd /d "%~dp0"

python -c "from orchestrator.llm import get_provider; p=get_provider(); print('Provider:', p.name); print('Model:', p.model)"
if errorlevel 1 exit /b 1
if /I "%~1"=="--check" exit /b 0

set "ORCHESTRATOR_PORT=%~1"
if "%ORCHESTRATOR_PORT%"=="" set "ORCHESTRATOR_PORT=8000"

echo Dashboard: http://127.0.0.1:%ORCHESTRATOR_PORT%/
python -m orchestrator.cli serve --port "%ORCHESTRATOR_PORT%"
exit /b %errorlevel%
