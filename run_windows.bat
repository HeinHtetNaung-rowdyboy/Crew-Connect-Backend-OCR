@echo off
cd /d "%~dp0"
if not exist .venv (py -3 -m venv .venv)
call .venv\Scripts\activate.bat
pip install -r requirements.txt
if not exist .env copy .env.example .env
start "Maritime FastAPI" cmd /k ".venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000"
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
