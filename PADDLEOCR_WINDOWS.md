# PaddleOCR Windows upgrade

This version adds PaddleOCR 3.x to the existing PyMuPDF → OCR → classifier → Qwen Vision → validator flow.

## Changed files
- `app/config.py`: `OCR_ENGINE`, `OCR_FALLBACK`, `PADDLEOCR_LANG`, `OCR_MIN_TEXT`
- `app/ocr/pdf_parser.py`: PaddleOCR 3.x `predict()` with Tesseract fallback
- `app/pipeline.py`: reports `processing.ocr_engines_used`
- `.env.example`: default PaddleOCR settings
- `requirements-paddleocr.txt`: optional dependency additions
- `tests/test_paddleocr.py`: routing and fallback tests

## Windows 10/11 setup (CMD in VS Code)

```cmd
cd C:\maritime-ai
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install paddlepaddle
pip install -r requirements-paddleocr.txt
copy .env.example .env
```

If you have an existing `.env`, edit it rather than overwriting it. Verify compatibility and installation instructions at https://www.paddlepaddle.org.cn/install/quick.

Ensure Tesseract is also installed if you want the fallback. Ollama remains optional and independent.

Start using `run_windows.bat`; the API and Streamlit UI have not changed.

## Environment keys
```env
OCR_BACKEND=local
OCR_ENGINE=paddleocr
OCR_FALLBACK=tesseract
PADDLEOCR_LANG=en
TESSERACT_LANG=eng
USE_OLLAMA=true
OLLAMA_MODEL=qwen2.5vl:3b
VISION_MAX_PAGES=10
```

The first PaddleOCR inference downloads its model files. In offline environments pre-cache them. For pure digital PDFs, the PDF text layer is used directly. When a page has an embedded image or limited text, OCR runs. If PaddleOCR fails/returns empty text, the Tesseract fallback is tried. `OCR_BACKEND` is intentionally still `local`, not `paddleocr`.

PaddleOCR has NOT been installed or benchmarked in the packaging environment. Validate both CPU runtime and field extraction on your Windows computer. Large documents may be slow; check page and upload limits.
