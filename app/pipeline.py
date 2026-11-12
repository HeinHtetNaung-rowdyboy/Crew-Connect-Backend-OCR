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


def _attempt(path: Path, engine, mode, earlier_cls=None):
    """One OCR engine -> classify -> rules -> validate -> gate."""
    started = time.perf_counter()
    parsed = cloud_parse(path) if mode == 'landingai' else local_parse(path, engine)
    classification = classify(parsed['text'])
    disagreement = False
    if earlier_cls:
        confident = classification['certificate_name'] != 'Unknown' and not classification['requires_review']
        if not confident:
            classification = earlier_cls  # the cheaper engine read the title better
        elif classification['certificate_name'] != earlier_cls['certificate_name']:
            disagreement = True
    fields = extract_fields(parsed['text'], classification['certificate_name'])
    data, validation_warnings = validate_and_normalize(classification, fields)
    ok, reasons = gate_check(classification, data, parsed['text'], engine)
    if disagreement:
        ok, reasons = False, reasons + ['OCR engines disagree on certificate type']
    return {'engine': engine or parsed['engine'], 'parsed': parsed, 'classification': classification,
            'fields': fields, 'disagreement': disagreement, 'ok': ok, 'reasons': reasons,
            'seconds': round(time.perf_counter() - started, 2)}


def _score(attempt):
    # Fewer gate failures is better; an unknown document type is much worse.
    return len(attempt['reasons']) + (10 if attempt['classification']['certificate_name'] == 'Unknown' else 0)


def _same(key, a, b):
    if key.endswith('_date'):
        return parse_date(a) == parse_date(b)
    return str(a).strip().casefold() == str(b).strip().casefold()


def process_document(path: Path, use_ollama: bool | None = None) -> dict:
    cfg = settings(); mode = cfg['backend']; warnings = []
    if mode not in ('local', 'landingai', 'hybrid'): raise ValueError('Invalid OCR_BACKEND')

    # ---- 1. OCR cascade: cheapest engine first, escalate only if the gate fails
    engines = cfg['ocr_cascade'] if mode != 'landingai' and cfg['ocr_cascade'] else [None]
    attempts, chosen, earlier_cls, any_disagreement = [], None, None, False
    for engine in engines:
        try:
            att = _attempt(path, engine, mode, earlier_cls)
        except ValueError:
            raise  # encrypted / too many pages: not an engine problem
        except Exception as exc:
            attempts.append({'engine': engine, 'passed': False, 'reasons': [], 'seconds': 0,
                             'error': f'{type(exc).__name__}: {str(exc)[:240]}'})
            continue
        any_disagreement = any_disagreement or att['disagreement']
        c = att['classification']
        if c['certificate_name'] != 'Unknown' and not c['requires_review']: earlier_cls = c
        attempts.append({'engine': att['engine'], 'passed': att['ok'], 'reasons': att['reasons'],
                         'seconds': att['seconds'], 'error': None})
        if chosen is None or _score(att) <= _score(chosen): chosen = att
        if att['ok']: break
    if chosen is None:
        raise RuntimeError('OCR failed in every engine: ' + '; '.join(str(a['engine']) + ': ' + a['error'] for a in attempts))
    for a in attempts:
        if a.get('error'): warnings.append(f"OCR engine {a['engine']} failed: {a['error']}")

    parsed, classification, fields, gate_ok = chosen['parsed'], chosen['classification'], chosen['fields'], chosen['ok']
    used_engine = chosen['engine']

    # ---- 2. Optional cloud paths (unchanged behaviour)
    if mode == 'hybrid' and classification['requires_review'] and cfg['landingai_key']:
        try:
            att = _attempt(path, None, 'landingai')
            parsed, classification, fields, gate_ok = att['parsed'], att['classification'], att['fields'], att['ok']
            used_engine = None
        except Exception as exc: warnings.append('Cloud fallback failed: ' + type(exc).__name__)
    if cfg['use_ai_extraction'] and parsed['engine'] == 'landingai':
        try:
            supplements = cloud_extract(parsed['text'])
            for k in fields:
                if not fields[k] and supplements.get(k): fields[k] = supplements[k]
        except Exception as exc: warnings.append('Cloud extraction failed: ' + type(exc).__name__)

    # ---- 3. Qwen only when the gate still fails; it may fill gaps, never overwrite
    ai_enabled = cfg['use_ollama'] if use_ollama is None else use_ollama
    run_ai = ai_enabled and not gate_ok
    reviewer = {'status': 'NOT_RUN', 'issues': []}
    conflicts, imgs = [], None
    if run_ai:
        from app.extraction.vision import extract_vision, render_pages
        try:
            imgs = render_pages(path)
            if cfg['use_llm_classifier'] and classification['requires_review']:
                from app.classification.llm_classifier import llm_classify
                proposal = llm_classify(parsed['text'], imgs)
                if proposal['certificate_name'] != 'Unknown': classification = proposal
            prediction = extract_vision(path, parsed['text'], classification['certificate_name'], imgs)
            for key, value in prediction.items():
                if not value: continue
                if not fields.get(key): fields[key] = value
                elif not _same(key, fields[key], value): conflicts.append(key)
        except Exception as exc:
            warnings.append('Ollama extraction failed; fallback to OCR rules: ' + f'{type(exc).__name__}: {str(exc)[:200]}')
    for key in conflicts:
        warnings.append(f'AI and rules disagree on {key}; kept the rule value, needs human review')

    # ---- 4. Final validation + gate (also grounds any AI-filled value in the OCR text)
    data, validation_warnings = validate_and_normalize(classification, fields)
    warnings.extend(validation_warnings)
    final_ok, final_reasons = gate_check(classification, data, parsed['text'], used_engine)
    if any_disagreement:
        final_ok, final_reasons = False, final_reasons + ['OCR engines disagree on certificate type']

    if run_ai and cfg['use_ai_reviewer']:
        try:
            from app.validation.reviewer import review
            reviewer = review(parsed['text'], data, imgs)
        except Exception as exc: reviewer = {'status': 'FAIL', 'issues': ['AI review unavailable: ' + type(exc).__name__]}

    deterministic_fail = bool(validation_warnings) or classification['requires_review'] or not final_ok or bool(conflicts)
    if reviewer['status'] == 'PASS' and deterministic_fail:
        reviewer = {'status': 'FAIL', 'issues': reviewer['issues'] + ['Rule validation requires review']}
    needs_review = deterministic_fail or reviewer['status'] == 'FAIL' or (run_ai and cfg['use_ai_reviewer'] and reviewer['status'] != 'PASS')

    cert = find_certificate(classification['certificate_name'])
    return {'success': True, 'data': data,
            'certificate_id': cert['id'] if cert else None,
            'classification': {'confidence': classification['confidence'], 'matched_aliases': classification['matched_aliases']},
            'validation': reviewer, 'requires_review': needs_review,
            'review_reasons': final_reasons, 'warnings': warnings,
            'processing': {'engine': parsed['engine'], 'page_count': len(parsed['pages']),
                           'qwen_enabled': ai_enabled, 'ai_used': run_ai,
                           'ocr_engines_used': parsed.get('ocr_engines_used', []),
                           'ocr_attempts': attempts}}