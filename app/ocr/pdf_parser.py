"""Extract PDF text with PyMuPDF and configurable OCR for scanned pages."""

import time
from pathlib import Path
from functools import lru_cache

import pymupdf as fitz

from app.config import settings


# --------------------------------------------------
# PaddleOCR initialization
# --------------------------------------------------

@lru_cache(maxsize=4)
def _make_paddle(language: str):
    """
    Initialize and cache PaddleOCR.

    enable_mkldnn=False disables oneDNN/MKLDNN
    to work around PaddlePaddle CPU inference errors.
    """

    from paddleocr import PaddleOCR

    cfg = settings()
    kwargs = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        text_det_limit_type="max",
        text_det_limit_side_len=cfg["paddle_det_side"],
    )

    # Explicit model names make PaddleOCR ignore `lang`.
    if cfg["paddle_det_model"]:
        kwargs["text_detection_model_name"] = cfg["paddle_det_model"]
    if cfg["paddle_rec_model"]:
        kwargs["text_recognition_model_name"] = cfg["paddle_rec_model"]
    if not (cfg["paddle_det_model"] or cfg["paddle_rec_model"]):
        kwargs["lang"] = language
    if cfg["paddle_cpu_threads"] > 0:
        kwargs["cpu_threads"] = cfg["paddle_cpu_threads"]

    return PaddleOCR(**kwargs)


def warmup():
    """Load the PaddleOCR model once at startup so the first request is not slow."""
    conf = settings()
    if "paddleocr" in conf["ocr_cascade"] or conf["ocr_engine"] == "paddleocr":
        from PIL import Image
        _paddle_ocr(Image.new("RGB", (320, 120), "white"), conf["paddleocr_lang"])


# --------------------------------------------------
# PaddleOCR text extraction
# --------------------------------------------------

def _paddle_ocr(image, lang: str) -> str:
    """
    Recognize text using PaddleOCR 3.x.

    Supports PaddleOCR result objects containing
    recognized text in the 'rec_texts' field.
    """

    import numpy as np

    ocr = _make_paddle(lang)

    image_array = np.asarray(image.convert("RGB"))

    results = ocr.predict(image_array)

    lines = []

    for item in results:

        if hasattr(item, "json"):
            data = item.json

            if callable(data):
                data = data()

        else:
            data = item

        if isinstance(data, dict):

            # PaddleOCR 3.x may wrap output in 'res'
            data = data.get("res", data)

            texts = data.get("rec_texts", [])

            for text in texts:
                if text:
                    lines.append(str(text))

    return "\n".join(lines).strip()


# --------------------------------------------------
# Tesseract fallback
# --------------------------------------------------

def _tesseract_ocr(image, lang: str) -> str:
    """
    Extract text using Tesseract OCR.
    """

    import pytesseract

    return pytesseract.image_to_string(
        image,
        lang=lang
    ).strip()


# --------------------------------------------------
# OCR engine selection
# --------------------------------------------------

def ocr_image(image, conf: dict):
    """
    Use the configured OCR engine.

    Primary:
        PaddleOCR or Tesseract

    Fallback:
        Tesseract or PaddleOCR
    """

    selected = conf.get(
        "ocr_engine",
        "tesseract"
    ).lower()

    fallback = conf.get(
        "ocr_fallback",
        "tesseract"
    ).lower()

    engines = [selected]

    if fallback not in (
        selected,
        "none",
        ""
    ):
        engines.append(fallback)

    failures = []

    for engine in engines:

        try:

            if engine == "paddleocr":

                result = _paddle_ocr(
                    image,
                    conf.get("paddleocr_lang", "en")
                )

            elif engine == "tesseract":

                result = _tesseract_ocr(
                    image,
                    conf.get("tesseract_lang", "eng")
                )

            else:
                raise ValueError(
                    "Unknown OCR_ENGINE: " + engine
                )

            if result.strip():

                return result.strip(), engine

            failures.append(
                f"{engine}: returned no text"
            )

        except Exception as exc:

            failures.append(
                f"{engine}: "
                f"{type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        "OCR failed: " + "; ".join(failures)
    )


# --------------------------------------------------
# PDF processing
# --------------------------------------------------

def parse_pdf(path: Path, engine: str | None = None) -> dict:
    """
    Extract text from a PDF.

    1. Read embedded text using PyMuPDF.
    2. Run OCR for scanned or sparse pages.
    3. Use PaddleOCR as primary engine if configured.
    4. Use Tesseract as fallback if necessary.
    """

    conf = settings()

    # Cascade mode: run exactly one engine, no hidden fallback.
    if engine:
        conf = {**conf, "ocr_engine": engine, "ocr_fallback": "none"}

    zoom = (
        conf["paddle_zoom"]
        if conf["ocr_engine"] == "paddleocr"
        else conf["tesseract_zoom"]
    )

    pages = []

    engines_used = set()

    ocr_seconds = 0.0

    last_failure = None

    ocr_attempted = 0

    ocr_succeeded = 0

    with fitz.open(path) as pdf:

        if pdf.needs_pass:

            raise ValueError(
                "Encrypted PDF is not supported in local mode"
            )

        if len(pdf) > conf["max_pages"]:

            raise ValueError(
                f"PDF exceeds maximum "
                f"{conf['max_pages']} pages"
            )

        for i, page in enumerate(pdf):

            # First try embedded PDF text
            content = page.get_text(
                "text",
                sort=True
            ).strip()

            # Determine whether OCR is needed
            # Any page with an image is OCR'd: scanned PDFs often carry a
            # garbage text layer that looks long but is unusable.
            requires_ocr = (
                len(content) < conf["ocr_min_text"]
                or bool(page.get_images(full=True))
            )

            if requires_ocr:

                from PIL import Image

                # Render PDF page as image
                from app.ocr.regions import page_clip, render

                clip = (
                    page_clip(page, conf["crop_max_area"], conf["crop_max_outside"])
                    if conf["autocrop"]
                    else None
                )

                pix = render(page, zoom, conf["ocr_crop_px"], clip)

                image = Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples
                )

                started = time.perf_counter()

                ocr_attempted += 1

                try:

                    ocr_text, used = ocr_image(
                        image,
                        conf
                    )

                    ocr_seconds += time.perf_counter() - started

                    if ocr_text:

                        ocr_succeeded += 1

                        # Combine extracted PDF text and OCR text
                        content = (
                            content + "\n" + ocr_text
                        ).strip()

                        engines_used.add(used)

                except RuntimeError as exc:

                    ocr_seconds += time.perf_counter() - started

                    # One blank/unreadable page must not fail the whole
                    # document; it is only an error if NO page has text.
                    last_failure = exc

            pages.append({
                "page": i + 1,
                "text": content
            })
    
    # A blank page is tolerated, but an engine that read NOTHING (crash,
    # missing model) must fail loudly instead of passing on junk text layers.
    if ocr_attempted and not ocr_succeeded and last_failure is not None:
        raise last_failure

    return {
        "text": "\n\n".join(
            item["text"] for item in pages
        ),
        "pages": pages,
        "engine": "local",
        "ocr_engines_used": sorted(
            engines_used
        ),
        "ocr_seconds": round(ocr_seconds, 2)
    }