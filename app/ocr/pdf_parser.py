
"""Extract PDF text with PyMuPDF and configurable OCR for scanned pages."""

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

    return PaddleOCR(
        lang=language,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False
    )


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

def parse_pdf(path: Path) -> dict:
    """
    Extract text from a PDF.

    1. Read embedded text using PyMuPDF.
    2. Run OCR for scanned or sparse pages.
    3. Use PaddleOCR as primary engine if configured.
    4. Use Tesseract as fallback if necessary.
    """

    conf = settings()

    pages = []

    engines_used = set()

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
            requires_ocr = (
                len(content) < conf["ocr_min_text"]
                or bool(page.get_images(full=True))
            )

            if requires_ocr:

                from PIL import Image

                # Render PDF page as image
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )

                image = Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples
                )

                try:

                    ocr_text, engine = ocr_image(
                        image,
                        conf
                    )

                    if ocr_text:

                        # Combine extracted PDF text and OCR text
                        content = (
                            content + "\n" + ocr_text
                        ).strip()

                        engines_used.add(engine)

                except RuntimeError:

                    # Preserve embedded text when OCR fails
                    if not content:
                        raise

            pages.append({
                "page": i + 1,
                "text": content
            })
    
    return {
        "text": "\n\n".join(
            item["text"] for item in pages
        ),
        "pages": pages,
        "engine": "local",
        "ocr_engines_used": sorted(
            engines_used
        )
    }
