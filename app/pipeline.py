"""Maritime certificate pipeline with per-document, per-stage timings."""
from __future__ import annotations

import time
from pathlib import Path

from app.config import settings
from app.ocr.pdf_parser import parse_pdf as local_parse
from app.ocr.landingai import parse_pdf as cloud_parse, extract_fields as cloud_extract
from app.classification.classifier import classify
from app.extraction.extractor import extract_fields
from app.extraction.date_parser import parse_date
from app.validation.validator import validate_and_normalize
from app.validation.gate import check as gate_check, find_certificate


def _attempt(path: Path, engine, mode: str, earlier_cls=None) -> dict:
    """One OCR attempt: parse, classify, extract/validate, and quality gate."""
    started = time.perf_counter()
    stages = {"ocr_s": 0.0, "classification_s": 0.0, "rules_validation_s": 0.0}

    try:
        t = time.perf_counter()
        try:
            parsed = cloud_parse(path) if mode == "landingai" else local_parse(path, engine)
        finally:
            stages["ocr_s"] += time.perf_counter() - t

        t = time.perf_counter()
        try:
            classification = classify(parsed["text"])
            disagreement = False
            if earlier_cls:
                confident = (
                    classification["certificate_name"] != "Unknown"
                    and not classification["requires_review"]
                )
                if not confident:
                    classification = earlier_cls
                elif classification["certificate_name"] != earlier_cls["certificate_name"]:
                    disagreement = True
        finally:
            stages["classification_s"] += time.perf_counter() - t

        t = time.perf_counter()
        try:
            fields = extract_fields(parsed["text"], classification["certificate_name"])
            data, validation_warnings = validate_and_normalize(classification, fields)
            ok, reasons = gate_check(classification, data, parsed["text"], engine)
            if disagreement:
                ok, reasons = False, reasons + ["OCR engines disagree on certificate type"]
        finally:
            stages["rules_validation_s"] += time.perf_counter() - t

        return {
            "engine": engine or parsed["engine"],
            "parsed": parsed,
            "classification": classification,
            "fields": fields,
            "disagreement": disagreement,
            "ok": ok,
            "reasons": reasons,
            "seconds": round(time.perf_counter() - started, 3),
            "stages": stages,
        }
    except Exception as exc:
        # Attach timing even when an OCR engine throws, so failed fallback calls count.
        exc._maritime_stages = stages
        exc._maritime_elapsed = time.perf_counter() - started
        raise


def _score(attempt: dict) -> int:
    return len(attempt["reasons"]) + (
        10 if attempt["classification"]["certificate_name"] == "Unknown" else 0
    )


def _same(key, a, b) -> bool:
    if key.endswith("_date"):
        return parse_date(a) == parse_date(b)
    return str(a).strip().casefold() == str(b).strip().casefold()


def process_document(path: Path, use_ollama: bool | None = None) -> dict:
    """Process one PDF. Images are converted to PDFs by the upload API."""
    total_started = time.perf_counter()
    cfg = settings()
    mode = cfg["backend"]
    if mode not in ("local", "landingai", "hybrid"):
        raise ValueError("Invalid OCR_BACKEND")

    warnings = []
    timings = {
        "ocr_s": 0.0,
        "classification_s": 0.0,
        "rules_validation_s": 0.0,
        "vision_render_s": 0.0,
        "llm_classifier_s": 0.0,
        "vision_inference_s": 0.0,
        "ai_reviewer_s": 0.0,
        "cloud_extraction_s": 0.0,
    }

    def add_stages(stages: dict):
        for key in ("ocr_s", "classification_s", "rules_validation_s"):
            timings[key] += stages.get(key, 0.0)

    engines = (
        cfg["ocr_cascade"] if mode != "landingai" and cfg["ocr_cascade"] else [None]
    )
    attempts, chosen, earlier_cls, any_disagreement = [], None, None, False
    for engine in engines:
        try:
            att = _attempt(path, engine, mode, earlier_cls)
        except ValueError:
            raise
        except Exception as exc:
            add_stages(getattr(exc, "_maritime_stages", {}))
            attempts.append({
                "engine": engine,
                "passed": False,
                "reasons": [],
                "seconds": round(getattr(exc, "_maritime_elapsed", 0.0), 3),
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            })
            continue

        add_stages(att["stages"])
        any_disagreement = any_disagreement or att["disagreement"]
        c = att["classification"]
        if c["certificate_name"] != "Unknown" and not c["requires_review"]:
            earlier_cls = c
        attempts.append({
            "engine": att["engine"],
            "passed": att["ok"],
            "reasons": att["reasons"],
            "seconds": att["seconds"],
            "error": None,
        })
        if chosen is None or _score(att) <= _score(chosen):
            chosen = att
        if att["ok"]:
            break

    if chosen is None:
        raise RuntimeError(
            "OCR failed in every engine: "
            + "; ".join(f"{a['engine']}: {a['error']}" for a in attempts)
        )
    for att in attempts:
        if att.get("error"):
            warnings.append(f"OCR engine {att['engine']} failed: {att['error']}")

    parsed, classification, fields, gate_ok = (
        chosen["parsed"], chosen["classification"], chosen["fields"], chosen["ok"]
    )
    used_engine = chosen["engine"]

    if mode == "hybrid" and classification["requires_review"] and cfg["landingai_key"]:
        try:
            att = _attempt(path, None, "landingai")
            add_stages(att["stages"])
            parsed, classification, fields, gate_ok = (
                att["parsed"], att["classification"], att["fields"], att["ok"]
            )
            used_engine = None
        except Exception as exc:
            add_stages(getattr(exc, "_maritime_stages", {}))
            warnings.append("Cloud fallback failed: " + type(exc).__name__)

    if cfg["use_ai_extraction"] and parsed["engine"] == "landingai":
        t = time.perf_counter()
        try:
            supplements = cloud_extract(parsed["text"])
            for key in fields:
                if not fields[key] and supplements.get(key):
                    fields[key] = supplements[key]
        except Exception as exc:
            warnings.append("Cloud extraction failed: " + type(exc).__name__)
        finally:
            timings["cloud_extraction_s"] += time.perf_counter() - t

    ai_enabled = cfg["use_ollama"] if use_ollama is None else use_ollama
    run_ai = ai_enabled and not gate_ok
    reviewer = {"status": "NOT_RUN", "issues": []}
    conflicts, imgs = [], None

    if run_ai:
        from app.extraction.vision import extract_vision, render_pages
        try:
            t = time.perf_counter()
            try:
                imgs = render_pages(path)
            finally:
                timings["vision_render_s"] += time.perf_counter() - t

            if cfg["use_llm_classifier"] and classification["requires_review"]:
                from app.classification.llm_classifier import llm_classify
                t = time.perf_counter()
                try:
                    proposal = llm_classify(parsed["text"], imgs)
                    if proposal["certificate_name"] != "Unknown":
                        classification = proposal
                finally:
                    timings["llm_classifier_s"] += time.perf_counter() - t

            t = time.perf_counter()
            try:
                prediction = extract_vision(
                    path, parsed["text"], classification["certificate_name"], imgs
                )
            finally:
                timings["vision_inference_s"] += time.perf_counter() - t

            for key, value in prediction.items():
                if not value:
                    continue
                if not fields.get(key):
                    fields[key] = value
                elif not _same(key, fields[key], value):
                    conflicts.append(key)
        except Exception as exc:
            warnings.append(
                "Ollama extraction failed; fallback to OCR rules: "
                + f"{type(exc).__name__}: {str(exc)[:200]}"
            )

    for key in conflicts:
        warnings.append(f"AI and rules disagree on {key}; kept the rule value, needs human review")

    t = time.perf_counter()
    try:
        data, validation_warnings = validate_and_normalize(classification, fields)
        warnings.extend(validation_warnings)
        final_ok, final_reasons = gate_check(classification, data, parsed["text"], used_engine)
        if any_disagreement:
            final_ok, final_reasons = False, final_reasons + [
                "OCR engines disagree on certificate type"
            ]
    finally:
        timings["rules_validation_s"] += time.perf_counter() - t

    if run_ai and cfg["use_ai_reviewer"]:
        t = time.perf_counter()
        try:
            from app.validation.reviewer import review
            reviewer = review(parsed["text"], data, imgs)
        except Exception as exc:
            reviewer = {
                "status": "FAIL",
                "issues": ["AI review unavailable: " + type(exc).__name__],
            }
        finally:
            timings["ai_reviewer_s"] += time.perf_counter() - t

    deterministic_fail = (
        bool(validation_warnings)
        or classification["requires_review"]
        or not final_ok
        or bool(conflicts)
    )
    if reviewer["status"] == "PASS" and deterministic_fail:
        reviewer = {
            "status": "FAIL",
            "issues": reviewer["issues"] + ["Rule validation requires review"],
        }
    needs_review = (
        deterministic_fail
        or reviewer["status"] == "FAIL"
        or (run_ai and cfg["use_ai_reviewer"] and reviewer["status"] != "PASS")
    )

    cert = find_certificate(classification["certificate_name"])
    timings["vision_s"] = (
        timings["vision_render_s"]
        + timings["llm_classifier_s"]
        + timings["vision_inference_s"]
        + timings["ai_reviewer_s"]
    )
    timings["total_s"] = time.perf_counter() - total_started
    # Overhead is not an extra stage: just uninstrumented time within the total.
    instrumented = sum(
        value for key, value in timings.items()
        if key not in ("vision_s", "total_s", "other_s")
    )
    timings["other_s"] = max(0.0, timings["total_s"] - instrumented)
    timings = {key: round(value, 3) for key, value in timings.items()}

    return {
        "success": True,
        "data": data,
        "certificate_id": cert["id"] if cert else None,
        "classification": {
            "confidence": classification["confidence"],
            "matched_aliases": classification["matched_aliases"],
        },
        "validation": reviewer,
        "requires_review": needs_review,
        "review_reasons": final_reasons,
        "warnings": warnings,
        "processing": {
            "engine": parsed["engine"],
            "page_count": len(parsed["pages"]),
            "qwen_enabled": ai_enabled,
            "ai_used": run_ai,
            "ocr_engines_used": parsed.get("ocr_engines_used", []),
            "ocr_attempts": attempts,
            "timings": timings,
        },
    }
