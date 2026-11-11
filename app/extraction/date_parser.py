import re
from datetime import datetime

DATE_PATTERN = re.compile(r'(?<!\d)(\d{1,2}[./\-]\d{1,2}[./\-]\d{4}|\d{1,2}[\s\-]+[A-Za-z]{3,9}[\s,\-]+\d{4}|\d{4}-\d{2}-\d{2})(?!\d)')
FORMATS = ('%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y', '%d %b %Y', '%d-%b-%Y', '%d %B %Y', '%d-%B-%Y', '%Y-%m-%d')


def parse_date(value):
    if not value:
        return None
    value = re.sub(r'\s+', ' ', str(value)).strip().replace(',', '')
    for fmt in FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def first_date(text):
    for match in DATE_PATTERN.finditer(text):
        iso = parse_date(match.group(0))
        if iso:
            return iso
    return None
