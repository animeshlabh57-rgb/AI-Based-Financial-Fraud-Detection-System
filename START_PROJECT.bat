@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing_python
if exist "models\model_metadata.json" goto dashboard
".venv\Scripts\python.exe" main.py
if errorlevel 1 goto failed
:dashboard
".venv\Scripts\python.exe" -m streamlit run dashboard/app.py --server.address localhost
goto end
:missing_python
echo Create .venv and install requirements.txt using the README instructions.
pause
exit /b 1
:failed
echo Training failed. Read the error above.
pause
exit /b 1
:end
