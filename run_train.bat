@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "choquet_anfis\main.py" > train_run.out.log 2> train_run.err.log
