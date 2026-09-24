@echo off
REM Starts the web app at http://127.0.0.1:5000 and opens your browser.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run - performing setup...
  call "%~dp0setup.bat"
)
.venv\Scripts\python run.py
pause
