from unittest.mock import patch
from pathlib import Path
import app.pipeline as pl
from app.config import settings as real_settings
from app.validation.gate import check, date_in_text, license_in_text, required_fields

GOOD = {'start_date': '2023-11-21', 'end_date': '2028-10-25', 'license_number': 'ERP075076'}
CLS = {'certificate_name': 'All in One', 'requires_review': False}
TEXT = 'Date of issue 21 Nov 2023 expiry 25-10-2028 certificate ERP075076'


def test_gate_passes_complete_grounded_result():
    assert check(CLS, GOOD, TEXT) == (True, [])


def test_gate_flags_ungrounded_and_missing():
    ok, why = check(CLS, {**GOOD, 'license_number': 'XX999999'}, TEXT)
    assert not ok and 'license_number not found in OCR text' in why
    ok, why = check(CLS, {**GOOD, 'end_date': None}, TEXT)
    assert 'missing end_date' in why


def test_gate_license_format_for_passport():
    cls = {'certificate_name': '여권/Passport', 'requires_review': False}
    ok, why = check(cls, {'start_date': '2024-01-18', 'end_date': '2029-01-17', 'license_number': 'MI 4496'},
                    '18 Jan 2024 17 Jan 2029 MI 4496')
    assert not ok and 'license_number does not match expected format' in why


def test_covid_does_not_require_license():
    assert 'license_number' not in required_fields('COVID Vaccine')


def test_helpers():
    assert date_in_text('2023-11-21', '21 NOV 2023') and not date_in_text('2023-11-22', '21 NOV 2023')
    assert license_in_text('MES/HSB 19-194/2026', 'ref MES / HSB 19-194 / 2026')


def _fake_parse(texts):
    calls = []
    def parse(path, engine=None):
        calls.append(engine)
        return {'text': texts[engine], 'pages': [{}], 'engine': 'local', 'ocr_engines_used': [engine]}
    return parse, calls


PSSR_OK = ('PREVENTION OF AND RESPONSE TO VIOLENCE AND HARASSMENT\nDate of issue 10.08.2026\nMES/HSB 19-194/2026')


def test_cascade_stops_after_cheap_engine_when_gate_passes():
    parse, calls = _fake_parse({'tesseract': PSSR_OK, 'paddleocr': PSSR_OK})
    with patch.object(pl, 'local_parse', parse), patch.object(pl, 'settings', lambda: {**real_settings(), 'ocr_cascade': ['tesseract', 'paddleocr']}):
        r = pl.process_document(Path('x.pdf'), False)
    assert calls == ['tesseract'] and r['certificate_id'] == 'pssr'
    assert r['processing']['ocr_attempts'][0]['passed']


def test_cascade_escalates_and_survives_engine_crash():
    parse, calls = _fake_parse({'tesseract': 'PREVENTION OF AND RESPONSE TO VIOLENCE AND HARASSMENT', 'paddleocr': None})
    def crashing(path, engine=None):
        if engine == 'paddleocr':
            calls.append(engine); raise RuntimeError('oneDNN crash')
        return parse(path, engine)
    with patch.object(pl, 'local_parse', crashing), patch.object(pl, 'settings', lambda: {**real_settings(), 'ocr_cascade': ['tesseract', 'paddleocr']}):
        r = pl.process_document(Path('x.pdf'), False)
    assert calls == ['tesseract', 'paddleocr']
    assert r['requires_review'] and any('paddleocr failed' in w for w in r['warnings'])


def test_covid_requires_stronger_engine():
    cls = {'certificate_name': 'COVID Vaccine', 'requires_review': False}
    data = {'start_date': '2022-04-21', 'end_date': None, 'license_number': None}
    text = '1st dose 21 Apr 2022'
    ok, why = check(cls, data, text, 'tesseract')
    assert not ok and 'needs paddleocr for this certificate type' in why
    assert check(cls, data, text, 'paddleocr') == (True, [])


def test_passport_number_ignores_replaced_passport():
    from app.extraction.extractor import extract_fields, extract_passport_number
    page2 = 'This Passport is issued in replacement of passport No. ME306282 Issued In Yangon'
    page1 = 'Passport No MI449665 Date of issue 18 JAN 2024\nMI449665<0MMR7303032M2901178<<<<<<<<<<<<<<<2'
    assert extract_passport_number(page1 + '\n' + page2) == 'MI449665'
    assert extract_passport_number(page2) is None          # never guess the old number
    assert extract_fields(page2, '여권/Passport')['license_number'] is None
    assert extract_passport_number('Passport No\nMI449665 Nationality') == 'MI449665'


def _run(texts):
    parse, calls = _fake_parse(texts)
    with patch.object(pl, 'local_parse', parse), patch.object(pl, 'settings', lambda: {**real_settings(), 'ocr_cascade': ['tesseract', 'paddleocr']}):
        return pl.process_document(Path('x.pdf'), False)


SID_TITLE = 'SEAFARERS IDENTITY DOCUMENT ILO CONVENTION NO 185 SID NO'


def test_unreadable_later_engine_keeps_earlier_classification():
    r = _run({'tesseract': SID_TITLE, 'paddleocr': 'gibberish'})
    assert r['certificate_id'] == 'sid'


def test_engines_disagreeing_on_type_forces_review():
    r = _run({'tesseract': SID_TITLE, 'paddleocr': PSSR_OK})
    assert r['requires_review'] and 'OCR engines disagree on certificate type' in r['review_reasons']


def test_engine_that_reads_nothing_is_not_treated_as_success(tmp_path):
    from app.ocr import pdf_parser
    from unittest.mock import patch as p
    import pytest
    pdf = Path(__file__).resolve().parent.parent / 'data/samples/pssr/sample_001.pdf'
    if not pdf.exists():
        pytest.skip('sample missing')
    with p.object(pdf_parser, '_paddle_ocr', side_effect=RuntimeError('oneDNN crash')):
        with pytest.raises(RuntimeError):
            pdf_parser.parse_pdf(pdf, 'paddleocr')


def test_document_box_ignores_specks_and_finds_small_card():
    from PIL import Image, ImageDraw
    from app.ocr.regions import document_box
    page = Image.new('RGB', (600, 800), 'white')
    draw = ImageDraw.Draw(page)
    draw.rectangle((100, 60, 360, 220), fill=(40, 70, 120))            # the card
    draw.rectangle((0, 0, 230, 6), fill=(90, 90, 90))                   # scan-edge smudge
    for x, y in ((500, 400), (50, 700), (560, 90)):                     # dust
        draw.ellipse((x, y, x + 4, y + 4), fill='black')
    (x0, y0, x1, y1), _ = document_box(page)
    assert 0.10 < x0 < 0.2 and 0.05 < y0 < 0.1 and 0.55 < x1 < 0.7 and 0.25 < y1 < 0.35


def test_full_page_document_is_not_cropped():
    from PIL import Image, ImageDraw
    from app.ocr.regions import document_box
    page = Image.new('RGB', (600, 800), 'white')
    ImageDraw.Draw(page).rectangle((10, 10, 590, 790), fill=(90, 90, 90))
    (x0, y0, x1, y1), _ = document_box(page)
    assert (x1 - x0) * (y1 - y0) > 0.9


def test_license_grounding_tolerates_ocr_lookalikes():
    assert license_in_text('MI449665', 'Passport No M1449665 Nationality')
    assert license_in_text('MS0095000', 'MsOO95OOO')
    assert not license_in_text('MI449666', 'Passport No M1449665')


def test_vision_prompt_has_passport_rule():
    from app.extraction.vision import CERTIFICATE_EXTRACTION_RULES as rules
    assert 'replacement of passport' in rules and 'Passport' in rules