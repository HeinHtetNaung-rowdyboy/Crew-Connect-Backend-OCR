"""Unified maritime certificate uploads.

POST /api/v1/certificates/process
Multipart: file (single), files (repeated), use_ollama, forward_to_spring.
Returns simplified JSON: one object for a single document; list for multiple.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import zipfile

import fitz
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.pipeline import process_document

logger = logging.getLogger("uvicorn.error")
DOC_EXTS = {".pdf", ".jpg", ".jpeg", ".png", ".webp"}
UPLOAD_EXTS = DOC_EXTS | {".zip"}
MAX_FILES = int(os.getenv("MAX_BATCH_FILES", "100"))
MAX_ZIP_UNCOMPRESSED_MB = int(os.getenv("MAX_ZIP_UNCOMPRESSED_MB", "200"))
MAX_ZIP_ENTRIES = int(os.getenv("MAX_ZIP_ENTRIES", "250"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        from app.ocr.pdf_parser import warmup
        await run_in_threadpool(warmup)
    except Exception as exc:
        logger.warning("OCR warm-up skipped: %s: %s", type(exc).__name__, exc)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Maritime Certificate AI - Unified Upload",
    version="3.0.0",
)

origins = [
    value.strip()
    for value in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "ocr_backend": settings()["backend"]}


def _assert_file_type(name: str, content: bytes) -> None:
    suffix = Path(name).suffix.lower()
    if suffix not in DOC_EXTS:
        raise ValueError(f"Unsupported document type: {suffix or '(missing extension)'}")
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF signature")
    else:
        try:
            with Image.open(io.BytesIO(content)) as img:
                img.verify()
                expected = {
                    ".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"
                }[suffix]
                if img.format != expected:
                    raise ValueError(f"Image content does not match {suffix} extension")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Invalid or unsupported image") from exc


def _image_to_pdf(content: bytes, path: Path) -> None:
    with Image.open(io.BytesIO(content)) as img:
        img.load()
        if img.width * img.height > 40_000_000:
            raise ValueError("Image resolution exceeds 40 megapixels")
        rgb = img.convert("RGB")
        out = io.BytesIO()
        rgb.save(out, format="PNG")
        width, height = rgb.size
    doc = fitz.open()
    try:
        page = doc.new_page(width=width, height=height)
        page.insert_image(page.rect, stream=out.getvalue())
        doc.save(str(path))
    finally:
        doc.close()


def _process_one(name: str, content: bytes, use_ollama: bool | None) -> dict:
    start = perf_counter()
    item = {"filename": name, "success": False}
    try:
        _assert_file_type(name, content)
        with TemporaryDirectory(prefix="maritime_doc_") as tmp:
            pdf_path = Path(tmp) / "document.pdf"
            if Path(name).suffix.lower() == ".pdf":
                pdf_path.write_bytes(content)
            else:
                _image_to_pdf(content, pdf_path)
            result = process_document(pdf_path, use_ollama)
        item.update(result)
        item["filename"] = name
    except Exception as exc:
        logger.exception("Document processing failed for %s", name)
        item.update({"success": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    item.setdefault("processing", {})
    timings = item["processing"].setdefault("timings", {})
    timings["document_elapsed_s"] = round(perf_counter() - start, 3)
    return item


def _unpack_zip(blob: bytes, archive_name: str, max_each: int) -> list[tuple[str, bytes]]:
    docs: list[tuple[str, bytes]] = []
    total_bytes = 0
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            entries = zf.infolist()
            if len(entries) > MAX_ZIP_ENTRIES:
                raise ValueError("ZIP contains too many entries")
            for info in entries:
                if info.is_dir():
                    continue
                path = Path(info.filename.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe ZIP member path")
                if Path(info.filename).suffix.lower() not in DOC_EXTS:
                    continue
                if info.flag_bits & 0x1:
                    raise ValueError("Encrypted ZIP members are not supported")
                if info.file_size > max_each:
                    raise ValueError(f"ZIP member exceeds per-file size limit: {info.filename}")
                total_bytes += info.file_size
                if total_bytes > MAX_ZIP_UNCOMPRESSED_MB * 1024 * 1024:
                    raise ValueError("ZIP exceeds total uncompressed size limit")
                if len(docs) >= MAX_FILES:
                    raise ValueError("Too many documents in upload")
                with zf.open(info) as member:
                    data = member.read(max_each + 1)
                if len(data) > max_each:
                    raise ValueError("ZIP member exceeds per-file size limit")
                docs.append((f"{archive_name}/{info.filename}", data))
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ValueError("Invalid ZIP archive") from exc
    if not docs:
        raise ValueError("ZIP contains no supported PDF or image documents")
    return docs


def _spring_payload(name: str, item: dict) -> dict:
    """Minimal response contract shared by React and Spring Boot."""
    data = item.get("data") or {}
    return {
        "filename": name,
        "success": bool(item.get("success")),
        "data": {
            "certificate_name": data.get("certificate_name"),
            "license_number": data.get("license_number"),
            "start_date": data.get("start_date"),
            "end_date": data.get("end_date"),
        },
    }


def _forward_result(payload: dict) -> dict:
    base_url = os.getenv("SPRING_BOOT_URL", "").rstrip("/")
    if not base_url:
        return {"sent": False, "error": "SPRING_BOOT_URL is not set"}
    callback = os.getenv("SPRING_BOOT_CALLBACK_PATH", "/api/ai/certificates/results")
    try:
        with httpx.Client(
            timeout=float(os.getenv("SPRING_BOOT_TIMEOUT", "30")),
            trust_env=False,
        ) as client:
            response = client.post(base_url + "/" + callback.lstrip("/"), json=payload)
            response.raise_for_status()
        return {"sent": True, "status_code": response.status_code}
    except httpx.HTTPError as exc:
        logger.exception("Spring Boot delivery failed")
        return {"sent": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


@app.post("/api/v1/certificates/process")
async def process_uploads(
    file: list[UploadFile] | None = File(default=None),
    files: list[UploadFile] | None = File(default=None),
    use_ollama: bool | None = Form(default=None),
    forward_to_spring: bool = Form(default=True),
):
    # Accept one or multiple uploads under either "file" or "files".
    uploads = (file or []) + (files or [])

    logger.info(
        "Received %d files: %s",
        len(uploads),
        [upload.filename for upload in uploads],
    )

    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="No files received. Use 'file' or 'files' as the multipart form-data key.",
        )
    if len(uploads) > MAX_FILES:
        raise HTTPException(status_code=413, detail="Too many uploaded files")

    max_each = int(settings()["max_upload_mb"]) * 1024 * 1024
    docs: list[tuple[str, bytes]] = []
    upload_errors: list[dict] = []
    for upload in uploads:
        name = Path(upload.filename or "").name
        suffix = Path(name).suffix.lower()
        if suffix not in UPLOAD_EXTS:
            upload_errors.append({"filename": name, "success": False, "error": "Unsupported upload extension"})
            continue
        cap = (
            MAX_ZIP_UNCOMPRESSED_MB if suffix == ".zip"
            else int(settings()["max_upload_mb"])
        ) * 1024 * 1024
        blob = await upload.read(cap + 1)
        if len(blob) > cap:
            upload_errors.append({"filename": name, "success": False, "error": "Upload exceeds size limit"})
            continue
        if suffix == ".zip":
            try:
                members = _unpack_zip(blob, name, max_each)
                if len(docs) + len(members) > MAX_FILES:
                    raise ValueError("Too many documents in upload")
                docs.extend(members)
            except ValueError as exc:
                upload_errors.append({"filename": name, "success": False, "error": str(exc)})
        else:
            docs.append((name, blob))
        if len(docs) > MAX_FILES:
            raise HTTPException(status_code=413, detail="Too many documents")

    results = list(upload_errors)
    for name, blob in docs:
        item = await run_in_threadpool(_process_one, name, blob, use_ollama)
        if forward_to_spring and item.get("success"):
            payload = _spring_payload(name, item)
            delivery = await run_in_threadpool(_forward_result, payload)
            if not delivery.get("sent"):
                logger.warning("Spring Boot delivery not completed for %s: %s", name, delivery.get("error"))
        results.append(item)

    simplified_results = [
        _spring_payload(item.get("filename", ""), item) for item in results
    ]
    if len(simplified_results) == 1:
        return simplified_results[0]
    return simplified_results
