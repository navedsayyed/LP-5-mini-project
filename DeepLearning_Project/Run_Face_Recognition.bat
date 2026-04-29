@echo off
cd /d "%~dp0"
call dl_env\Scripts\activate.bat
set PYTHONIOENCODING=utf-8
python dl.py
pause
