from contextlib import asynccontextmanager
from pathlib import Path
import logging
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.pipeline import process_document
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app):
    # Load PaddleOCR once at startup so the first request is not slow.
    try:
        from app.ocr.pdf_parser import warmup
        await run_in_threadpool(warmup)
    except Exception as exc:
        logger.warning("OCR warm-up skipped: %s: %s", type(exc).__name__, exc)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Maritime Certificate AI",
    version="2.0.0",
    description="Maritime classification, dates, OCR and optional Qwen vision"
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ocr_backend": settings()["backend"]
    }


@app.post("/api/v1/certificates/extract")
async def extract_certificate(
    file: UploadFile = File(...),
    use_ollama: bool | None = Form(default=None)
):
    # Validate uploaded file
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Upload a .pdf file"
        )

    # Maximum upload size
    limit = settings()["max_upload_mb"] * 1024 * 1024

    with TemporaryDirectory(prefix="maritime_") as folder:
        path = Path(folder) / "upload.pdf"
        size = 0

        # Save PDF
        with path.open("wb") as dest:
            while block := await file.read(1024 * 1024):
                size += len(block)

                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds upload limit"
                    )

                dest.write(block)

        # Check PDF signature
        with path.open("rb") as pdf:
            signature = pdf.read(5)

        if size < 5 or signature != b"%PDF-":
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF"
            )

        try:
            # OCR + classification + Qwen Vision extraction
            result = await run_in_threadpool(
                process_document,
                path,
                use_ollama
            )

            data = result.get("data") or {}

            # Return simplified JSON
            return {
                "success": result.get("success", False),
                "data": {
                    "certificate_type": data.get("certificate_type"),
                    "certificate_name": data.get("certificate_name"),
                    "expiration": data.get("expiration"),
                    "category": data.get("category"),
                    "start_date": data.get("start_date"),
                    "end_date": data.get("end_date"),
                    "license_number": data.get("license_number"),
                    "certificate_id": result.get("certificate_id")
                },
                "requires_review": result.get("requires_review"),
                "review_reasons": result.get("review_reasons", []),
                "warnings": result.get("warnings", [])
            }

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc)
            ) from exc

        
        except Exception as exc:
            logger.exception(
                "Maritime AI certificate extraction failed"
            )

            raise HTTPException(
                status_code=500,
                detail="Processing error: " + type(exc).__name__
            ) from exc