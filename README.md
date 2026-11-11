# Maritime AI v2 — Windows Streamlit + FastAPI + optional Qwen

This is an original, NanoNets DocAIAgent-inspired modular implementation (not a copied fork). It classifies maritime PDFs with configuration-based rules, OCRs scanned pages, extracts dates/numbers and optionally uses a local Ollama-hosted Qwen2.5-VL model for vision extraction and an AI reviewer. Not production-certified; always review sensitive documents and verify field accuracy.

## Windows 10/11 installation

1. Install **Python 3.11 or 3.12** (check "Add Python to PATH") and **Tesseract OCR for Windows** (UB Mannheim Windows installer). After installing, add the Tesseract directory (usually `C:\Program Files\Tesseract-OCR`) to the system PATH. Open a new Command Prompt and run `tesseract --version`.
2. Unzip this project. From Command Prompt, run `run_windows.bat`. On the first run it installs dependencies and creates `.env`.
3. FastAPI docs: http://127.0.0.1:8000/docs . Streamlit normally: http://localhost:8501 .
4. Upload one of the provided PDFs from `data/samples/` in Streamlit. Default local mode needs **no model/API key**.
5. For Qwen vision, install **Ollama for Windows** from https://ollama.com/download and run `ollama pull qwen2.5vl:3b` in a terminal. Ensure Ollama is serving `http://127.0.0.1:11434`. Select **Enable Qwen2.5-VL** in Streamlit. The 3B variant is easier to test on consumer hardware but may still require substantial RAM; larger models need more resources. First inference can take time. This mode is optional and is not validated for all documents.


ollama pull qwen2.5vl:3b

ollama run qwen2.5vl:3b



Manual commands on Windows:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload --port 8000
# separate terminal:
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## API

`POST /api/v1/certificates/extract` multipart `file` (PDF) and optional form `use_ollama=true/false`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/certificates/extract -F "file=@data/samples/pssr/sample_001.pdf" -F "use_ollama=false"
```

Response includes `data.start_date`, `data.end_date`, individual date fields, `data.expiration`, `classification`, `validation`, `requires_review`, and `warnings`.

**Start date priority:** training_date > issue_date > examination_date; first valid non-null value. **End date:** expire_date only. Expiration is `DateLimit` if end date present, otherwise `None`. Missing values are JSON null. A missing expiry is never calculated.

## Add the other ~135 certificate definitions

Modify `data/certificate_setup.json` and add unique aliases and distinctive `required_any` patterns; put representative PDFs under `data/samples/<type>/` and label verified fields in `data/labels/ground_truth.json`. Alias matching is not model training; evaluate classifications and extraction quality before production.

## AI reviewer and limitations

`app/llm/client.py` uses Ollama local REST API. `app/classification/llm_classifier.py` is used only for uncertain classifications. `app/extraction/vision.py` renders PDF pages (first `VISION_MAX_PAGES`) and requests visible date/number fields. `app/validation/reviewer.py` returns PASS/FAIL based on model response; **rule-validation errors override AI PASS**. In offline/local mode `validation.status=NOT_RUN`, not a fabricated PASS.

OCR on scanned passports, low-quality images and complex All-in-One date tables is imperfect; check all dates, especially the expiry date. A confidence score is heuristic, not calibrated. Don't treat generated output as verified evidence. The uploaded example PDFs contain personal data; do not commit them to a public Git repository. Delete them before sharing this project publicly.

## Docker (local OCR only by default)

```bash
docker build -t maritime-ai .
docker run --rm -p 8000:8000 maritime-ai
```

For Windows-hosted Ollama with Docker, use `OLLAMA_URL=http://host.docker.internal:11434` and enable the mode explicitly. Streamlit runs outside Docker by default.

## Testing

```bash
python -m pytest -q
```

## Credits

Modular architecture inspired by https://github.com/NanoNets/DocAIAgent . This project does not bundle NanoNets code, models, or cloud API credentials. Qwen2.5-VL is hosted through Ollama if installed separately.
