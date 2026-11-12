"""Qwen2.5-VL vision extraction via local Ollama with PDF rasterization."""

import pymupdf as fitz
from pathlib import Path

from app.config import settings
from app.llm.client import json_chat


FIELD_NAMES = [
    "training_date",
    "issue_date",
    "examination_date",
    "expire_date",
    "license_number"
]


# Certificate-specific extraction instructions
CERTIFICATE_EXTRACTION_RULES = """
Certificate-specific extraction rules:

1. COVID Vaccine
   - Find the vaccination table.
   - Locate the row labeled "1st Dose" or "First Dose".
   - Extract the date under the "Vaccination Date" column.
   - Do not use Appointment Date, 2nd Dose, or Booster Date.
   - Return the first-dose vaccination date as training_date.
   - If the 1st Dose vaccination date is not visible
     or is ambiguous, return null for training_date.

2. 선원수첩/Seaman Book (MM)
   - Locate the field labeled "Book No.",
     "Book Number", or "Seaman Book No.".
   - Extract its exact value.
   - Return that value as license_number.
   - Do not confuse Book No. with Passport No.,
     SID No., or SIRB references printed elsewhere.
   - If Book No. is missing or unclear,
     return null for license_number.

3. 여권/Passport
   - license_number is the CURRENT passport number, printed
     next to the label "Passport No" on the photo page and at the
     start of the second line of the machine-readable zone
     (e.g. MI449665<0MMR...).
   - Ignore any number in an endorsement or observation such as
     "issued in replacement of passport No. ...": that is an OLD
     passport.
   - Return the number exactly as printed; if unclear, return null.

General rules:
   - Extract only information visible in the document.
   - Never invent dates or numbers.
   - Do not use Date of Birth as a certificate date.
   - Do not calculate or infer expiration dates.
   - Preserve the original certificate number text.
   - Convert unambiguous dates to YYYY-MM-DD.
   - Return null when a value is missing or unclear.
   - Return only the requested JSON fields.
"""


def render_pages(path: Path) -> list[bytes]:
    cfg = settings()
    images = []

    with fitz.open(path) as pdf:
        if len(pdf) > cfg["max_pages"]:
            raise ValueError("Too many pages")

        from app.ocr.regions import page_clip, render

        for page in list(pdf)[:cfg["vision_max_pages"]]:
            clip = (
                page_clip(page, cfg["crop_max_area"], cfg["crop_max_outside"])
                if cfg["autocrop"]
                else None
            )
            pix = render(
                page,
                cfg["vision_zoom"],
                cfg["vision_crop_px"],
                clip
            )
            images.append(pix.tobytes("png"))

    return images


def extract_vision(
    path: Path,
    text: str,
    certificate_name: str,
    images=None
) -> dict:

    images = images if images is not None else render_pages(path)

    prompt = (
        f"Document classification: {certificate_name}.\n"
        "Read the attached maritime document pages.\n"
        "Extract ONLY explicitly visible values.\n\n"

        "Return a valid JSON object with exactly these keys:\n"
        "training_date, issue_date, examination_date, "
        "expire_date, license_number.\n\n"

        "Dates must be YYYY-MM-DD if unambiguous; "
        "otherwise return null.\n"
        "Do not use DOB as issue, training, or examination date.\n"
        "Do not infer expiry or invent a license number.\n\n"

        "For a multi-certificate booklet, report document-level "
        "dates only when consistent.\n"
        "If multiple distinct expiry dates exist and no single "
        "document-level expiry is clear, return null.\n\n"

        f"{CERTIFICATE_EXTRACTION_RULES}\n\n"

        "Original OCR context (may contain mistakes):\n"
        + text[:9000]
    )

    obj = json_chat(
        "You are a maritime document transcription worker. "
        "Return only valid JSON; use null for missing fields.",
        prompt,
        images
    )

    if not isinstance(obj, dict):
        raise ValueError(
            "Vision extractor returned non-object"
        )

    return {
        key: (
            obj.get(key)
            if isinstance(obj.get(key), str)
            else None
        )
        for key in FIELD_NAMES
    }