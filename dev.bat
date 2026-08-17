@echo off
set ROOT=%~dp0
set ROOT=%ROOT:~0,-1%

echo Checking Python dependencies...
call "%ROOT%\.venv\Scripts\activate.bat"
python -c "import fastapi" 2>nul || (
  echo Installing requirements...
  python -m pip install -r "%ROOT%\requirements.txt"
)

echo Starting FightPath locally...
start "FightPath API" cmd /k "cd /d "%ROOT%" && call .venv\Scripts\activate.bat && set PYTHONPATH=src && python -m uvicorn src.ufc.api.app:app --reload --port 8000"
timeout /t 3 /nobreak >nul
start "FightPath UI"  cmd /k "cd /d "%ROOT%\frontend" && npm run dev"
timeout /t 5 /nobreak >nul
start http://localhost:5173
