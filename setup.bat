@echo off
REM One-time setup: virtual environment, Python packages, headless Chromium.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || (echo Failed to create venv & pause & exit /b 1)
)
echo Installing packages...
.venv\Scripts\python -m pip install --upgrade pip >nul
.venv\Scripts\python -m pip install -r requirements.txt || (echo pip install failed & pause & exit /b 1)
echo Installing headless Chromium for the scraper...
.venv\Scripts\python -m playwright install chromium || (echo Chromium install failed - scraping will not work, the rest of the app will. & pause)
echo Training the model...
.venv\Scripts\python cli.py train
echo.
echo Setup complete. Double-click run.bat to start the app.
pause
