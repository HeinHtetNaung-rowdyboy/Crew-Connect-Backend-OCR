"""Deterministic validity gate.

Decides whether a rule-based result is trustworthy enough to skip the slower
steps (PaddleOCR, Qwen). It only reads the data and OCR text; it never calls
a model. A wrong value that passes this gate is NOT re-checked by anything
else, so keep it strict: when in doubt, return a reason.
"""
import re
from datetime import date, datetime

from app.classification.classifier import load_certificate_config

# Which normalized fields must exist for each certificate id.
# Override per certificate with "required_fields" in certificate_setup.json.
# Based on the 6 labeled samples: confirm against your real certificates.
DEFAULT_REQUIRED = {
    "all_in_one": ("start_date", "end_date", "license_number"),
    "pssr": ("start_date", "license_number"),
    "covid_vaccine": ("start_date",),
    "passport": ("start_date", "end_date", "license_number"),
    "sid": ("start_date", "end_date", "license_number"),
    "seaman_book": ("start_date", "end_date", "license_number"),
}
FALLBACK_REQUIRED = ("start_date", "license_number")
MAX_VALIDITY_YEARS = 30


def find_certificate(name):
    for item in load_certificate_config():
        if item["name"] == name:
            return item
    return None


def required_fields(name):
    cert = find_certificate(name)
    if not cert:
        return FALLBACK_REQUIRED
    return tuple(cert.get("required_fields") or DEFAULT_REQUIRED.get(cert["id"], FALLBACK_REQUIRED))


# OCR often swaps look-alike characters (MI449665 -> M1449665). Compare with
# them folded together so a correct value is not rejected for an OCR slip.
_LOOKALIKE = str.maketrans("oilsbz", "011582")


def _squash(text):
    return re.sub(r"[^a-z0-9]", "", str(text).casefold()).translate(_LOOKALIKE)


def license_in_text(value, text):
    needle = _squash(value)
    return bool(needle) and needle in _squash(text)


def date_in_text(iso, text):
    """True if the date appears in the OCR text in any common written form."""
    try:
        d = datetime.strptime(iso, "%Y-%m-%d")
    except (TypeError, ValueError):
        return False
    hay = re.sub(r"\s+", "", text.casefold())
    forms = {
        d.strftime("%Y-%m-%d"),
        d.strftime("%Y/%m/%d"),
        d.strftime("%Y.%m.%d"),
        d.strftime("%d %b %Y"), d.strftime("%d-%b-%Y"), d.strftime("%d %B %Y"),
        d.strftime("%d-%B-%Y"), f"{d.day} {d.strftime('%b')} {d.year}",
        f"{d.day} {d.strftime('%B')} {d.year}",
    }
    for sep in (".", "/", "-"):
        forms.add(f"{d.day:02d}{sep}{d.month:02d}{sep}{d.year}")
        forms.add(f"{d.day}{sep}{d.month}{sep}{d.year}")
    return any(re.sub(r"\s+", "", f.casefold()) in hay for f in forms)


def check(classification, data, text, engine=None):
    """Return (passed, reasons). `data` is the output of validate_and_normalize.

    `engine` is the OCR engine that produced `text`. A certificate can set
    "require_engine" in the config when a cheap engine is known to misread it.
    """
    reasons = []
    name = classification.get("certificate_name")
    cert = find_certificate(name)
    if name == "Unknown" or not cert:
        return False, ["unknown certificate type"]
    if classification.get("requires_review"):
        reasons.append("classification uncertain")
    needed = cert.get("require_engine")
    if needed and engine and engine != needed:
        reasons.append(f"needs {needed} for this certificate type")

    for field in required_fields(name):
        if not data.get(field):
            reasons.append(f"missing {field}")

    start, end = data.get("start_date"), data.get("end_date")
    if start and start > date.today().isoformat():
        reasons.append("start_date is in the future")
    if start and end:
        if end < start:
            reasons.append("end_date before start_date")
        elif int(end[:4]) - int(start[:4]) > MAX_VALIDITY_YEARS:
            reasons.append("validity period implausibly long")

    for field in ("start_date", "end_date"):
        if data.get(field) and not date_in_text(data[field], text):
            reasons.append(f"{field} not found in OCR text")

    lic = data.get("license_number")
    if lic:
        pattern = cert.get("license_pattern")
        compact = lic.strip()
        if pattern:
            if not re.fullmatch(pattern, compact):
                reasons.append("license_number does not match expected format")
        elif not (3 <= len(compact) <= 30 and re.search(r"\d", compact)):
            reasons.append("license_number format implausible")
        if not license_in_text(lic, text):
            reasons.append("license_number not found in OCR text")

    return not reasons, reasons