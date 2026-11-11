from datetime import date
from app.extraction.date_parser import parse_date
from app.extraction.schemas import CertificateData

DATE_FIELDS = ('training_date','issue_date','examination_date','expire_date')

def validate_and_normalize(classification: dict, fields: dict):
    warnings=[]
    clean={}
    for key in DATE_FIELDS:
        raw=fields.get(key)
        normalized=parse_date(raw)
        if raw and not normalized: warnings.append(f'Unrecognized {key}: {str(raw)[:60]}')
        clean[key]=normalized
    clean['start_date']=next((clean[key] for key in ('training_date','issue_date','examination_date') if clean[key]), None)
    clean['end_date']=clean['expire_date']
    clean['expiration']='DateLimit' if clean['end_date'] else 'None'
    clean['license_number']=str(fields['license_number']).strip() if fields.get('license_number') else None
    if clean['start_date'] and clean['end_date'] and clean['end_date'] < clean['start_date']:
        warnings.append('End date precedes start date')
    if clean['end_date'] and clean['end_date'] < date.today().isoformat():
        warnings.append('Document appears expired')
    if not clean['license_number']: warnings.append('License/certificate number not confidently extracted')
    if classification['certificate_name']=='Unknown': warnings.append('Unknown document type')
    document=CertificateData(certificate_name=classification['certificate_name'], certificate_type=classification['certificate_type'], category=classification['category'], **clean)
    return document.model_dump(),warnings
