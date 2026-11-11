Set-Location $PSScriptRoot
if (-not (Test-Path .venv)) { py -3 -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
Start-Process powershell -ArgumentList '-NoExit','-Command',"Set-Location '$PSScriptRoot'; & '.\.venv\Scripts\python.exe' -m uvicorn app.api:app --host 127.0.0.1 --port 8000"
& .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
