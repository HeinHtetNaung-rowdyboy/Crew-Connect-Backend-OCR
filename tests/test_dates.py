from app.validation.validator import validate_and_normalize

def base(): return {'certificate_name':'PSSR','certificate_type':'Additional Certificate','category':'CERTIFICATE'}

def test_start_priority_and_missing_expiry():
    d,w=validate_and_normalize(base(),{'training_date':'2026-08-10','issue_date':'2026-08-11','examination_date':'2026-08-09','expire_date':None,'license_number':'ABC-123'})
    assert d['start_date']=='2026-08-10' and d['end_date'] is None and d['expiration']=='None'

def test_fallback_issue_and_end_date():
    d,_=validate_and_normalize(base(),{'issue_date':'2024-01-01','expire_date':'2029-01-01','license_number':'ABC-123'})
    assert d['start_date']=='2024-01-01' and d['end_date']=='2029-01-01' and d['expiration']=='DateLimit'

def test_exam_fallback():
    d,_=validate_and_normalize(base(),{'examination_date':'2024-03-20','license_number':'ABC-123'})
    assert d['start_date']=='2024-03-20'
