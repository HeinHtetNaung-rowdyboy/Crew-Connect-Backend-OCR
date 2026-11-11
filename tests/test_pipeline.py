from pathlib import Path
from app.classification.classifier import classify
from app.extraction.date_parser import parse_date
from app.extraction.extractor import extract_fields
from app.validation.validator import validate_and_normalize
from app.pipeline import process_document
from app.config import ROOT


def test_date_formats():
    assert parse_date('10.08.2026') == '2026-08-10'
    assert parse_date('21-Nov-2023') == '2023-11-21'
    assert parse_date('2028-10-25') == '2028-10-25'


def test_classification():
    assert classify('PREVENTION OF AND RESPONSE TO VIOLENCE AND HARASSMENT')['certificate_name'] == 'PSSR'
    assert classify('Unknown unrelated document')['requires_review']
    assert classify('CERTIFICATE OF PROFICIENCY FOR SECURITY TRAINING safety familiarization and basic training')['certificate_name'] == 'All in One'


def test_expiration_logic():
    c = classify('PREVENTION OF AND RESPONSE TO VIOLENCE AND HARASSMENT')
    data, _ = validate_and_normalize(c, {'issue_date':'10.08.2026', 'expire_date': None})
    assert data['expiration'] == 'None'
    data, _ = validate_and_normalize(c, {'issue_date':'10.08.2026', 'expire_date': '25-Oct-2028'})
    assert data['expiration'] == 'DateLimit'


def test_sample_files():
    names = {'pssr': 'PSSR', 'all_in_one': 'All in One', 'passport': 'Passport (MM)', 'covid_vaccine': 'COVID Vaccine'}
    for folder, expected in names.items():
        path = ROOT/'data'/'samples'/folder/'sample_001.pdf'
        if path.exists():
            result = process_document(path)
            assert result['data']['certificate_name'] == expected, (folder, result)
